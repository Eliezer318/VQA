"""
Here, we create a custom dataset
"""
import torch
import pickle

from utils.types import PathT
from torch.utils.data import Dataset
from typing import Any, Tuple, Dict, List
from PIL import Image
import numpy as np

from tqdm import tqdm
import prep
import os
import json
import h5py


class Language:
    def __init__(self, json_path: str):
        self.word2idx = {'SOS': 0, 'EOS': 1, '<Unknown>': 2}
        self.idx2word = ['SOS', 'EOS', '<Unknown>']
        self.num_of_words = 3
        self.max_len = 11
        self.extract_words(json_path)

        # Questions Language
        self.label2ans = pickle.load(open('data/cache/trainval_label2ans.pkl', 'rb'))
        self.ans2label = pickle.load(open('data/cache/trainval_ans2label.pkl', 'rb'))

    @staticmethod
    def preprocess(txt):
        return prep.preprocess_answer(prep.process_punctuation(prep.process_digit_article(txt)))

    def extract_words(self, json_path):
        if os.path.exists('data/cache/idx2word.pkl'):
            self.word2idx = pickle.load(open('data/cache/word2idx.pkl', 'rb'))
            self.idx2word = pickle.load(open('data/cache/idx2word.pkl', 'rb'))
            self.num_of_words = len(self.idx2word)
            return None
        questions_json = json.load(open(json_path))['questions']
        for question_unit in tqdm(questions_json):
            question = Language.preprocess(question_unit['question'])
            for word in question.split(' '):
                if word not in self.word2idx.keys():
                    self.word2idx[word] = self.num_of_words
                    self.idx2word.append(word)
                    self.num_of_words += 1
        # dump the word2idx and the idx2word
        pickle.dump(self.word2idx, open('data/cache/word2idx.pkl', 'wb'))
        pickle.dump(self.idx2word, open('data/cache/idx2word.pkl', 'wb'))

    def txt2tensor(self, question_txt):
        question = Language.preprocess(question_txt).split(' ')
        word_unknown = lambda word: self.word2idx[word] if word in self.word2idx.keys() else self.word2idx['<Unknown>']
        question_indices = [0] + [word_unknown(word) for word in question] + [1]
        new_list = question_indices + max([self.max_len - len(question_indices), 0]) * [1, ]  # padding with EOS
        return torch.tensor(new_list[:self.max_len])
        # take care of case where len(txt) > self.max_sentence_len


class MyDataset(Dataset):

    @staticmethod
    def create_images_h5py(directory_path, train=True):
        path_images = 'data/train_images3.h5' if train else 'data/val_images3.h5'
        if os.path.exists(path_images):
            return path_images
        with h5py.File(path_images, "w-") as archive:
            for filename in tqdm((os.listdir(directory_path))):
                image_path = os.path.join(directory_path, filename)
                pil_img = Image.open(image_path).convert("RGB")
                img = np.array(pil_img.resize([64, 64]))
                img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
                archive.create_dataset(image_path, data=img)
        return path_images

    def __init__(self, image_dir, q_path, img_path, target_path, train=True) -> None:
        if not os.path.exists(target_path):
            prep.main()
        self.paths = {'target_path': target_path, 'questions_path': q_path, 'img_folder': img_path,
                      'images_h5py': MyDataset.create_images_h5py(image_dir, train),
                      'entries_path': 'data/cache/entries_train.pkl' if train else 'data/cache/entries_val.pkl',
                      'lang_path': 'data/cache/lang.pkl'
                      }

        if not os.path.exists(self.paths['lang_path']):
            self.train_lang = Language(self.paths['questions_path'])
            pickle.dump(self.train_lang, open(self.paths['lang_path'], 'wb'))
        else:
            self.train_lang = pickle.load(open(self.paths['lang_path'], 'rb'))
        self.entries = self._get_entries()
        self.total_len = 443757 if train else 214354

    def __getitem__(self, index: int) -> Tuple:
        image_path, question, label_counts = self.entries[index]
        img_tensor = h5py.File(self.paths['images_h5py'], "r")[image_path][:]
        answer = torch.zeros(len(self.train_lang.label2ans), dtype=torch.float32)
        for key in label_counts:
            answer[key] = prep.get_score(label_counts[key])
        max_label = -1
        if len(label_counts):
            max_label = max(label_counts, key=lambda x: label_counts[x])
        return img_tensor, question, answer, max_label

    def __len__(self) -> int:
        """
        :return: the length of the dataset (number of sample).
        """
        return self.total_len

    def _get_features(self) -> Any:
        features = pickle.load(open(self.paths['target_path'], "rb"))
        questions = json.load(open(self.paths['questions_path']))['questions']
        question_id2tensors = {qu['question_id']: self.train_lang.txt2tensor(qu['question']) for qu in questions}
        return features, question_id2tensors

    def _get_entries(self) -> List:
        """
        This function create a list of all the entries. We will use it later in __getitem__
        :return: list of samples
        """
        if os.path.exists(self.paths['entries_path']):
            return pickle.load(open(self.paths['entries_path'], 'rb'))
        feat, q2tensor = self._get_features()
        print('in entries list now')
        entry = lambda item: (self.id_to_path(item['image_id']), q2tensor[item['question_id']], item['label_counts'])
        entries = [entry(item) for item in tqdm(feat)]
        pickle.dump(entries, open(self.paths['entries_path'], 'wb'))
        return entries

    def id_to_path(self, image_id):
        return f'{self.paths["img_folder"]}_{(12 - len(str(image_id))) * "0" + str(image_id)}.jpg'
