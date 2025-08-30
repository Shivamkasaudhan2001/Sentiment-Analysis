{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4a7c81d8-e5e4-400f-8f5a-c321ad01f98b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import nltk\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "plt.style.use('ggplot')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "2f3ebc93-3192-4df6-be77-171e99a57741",
   "metadata": {},
   "outputs": [],
   "source": [
    "path ='/Users/mukulsaini/Downloads/archive/train.ft.txt'\n",
    "df = pd.read_fwf(path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4262d23e-f9cf-4ba4-bfc2-dba0b5cba9a2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>__label__2</th>\n",
       "      <th>Stuning even for the non-gamer: This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^</th>\n",
       "      <th>Unnamed: 2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>The best soundtrack ever to anything.: I'm rea...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>Amazing!: This soundtrack is my favorite music...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>Excellent Soundtrack: I truly like this soundt...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>Remember, Pull Your Jaw Off The Floor After He...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>an absolute masterpiece: I am quite sure any o...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>__label__1</td>\n",
       "      <td>Buyer beware: This is a self-published book, a...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>Glorious story: I loved Whisper of the wicked ...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>A FIVE STAR BOOK: I just finished reading Whis...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>Whispers of the Wicked Saints: This was a easy...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>__label__1</td>\n",
       "      <td>The Worst!: A complete waste of time. Typograp...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   __label__2  \\\n",
       "0  __label__2   \n",
       "1  __label__2   \n",
       "2  __label__2   \n",
       "3  __label__2   \n",
       "4  __label__2   \n",
       "5  __label__1   \n",
       "6  __label__2   \n",
       "7  __label__2   \n",
       "8  __label__2   \n",
       "9  __label__1   \n",
       "\n",
       "  Stuning even for the non-gamer: This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^  \\\n",
       "0  The best soundtrack ever to anything.: I'm rea...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "1  Amazing!: This soundtrack is my favorite music...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "2  Excellent Soundtrack: I truly like this soundt...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "3  Remember, Pull Your Jaw Off The Floor After He...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "4  an absolute masterpiece: I am quite sure any o...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "5  Buyer beware: This is a self-published book, a...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "6  Glorious story: I loved Whisper of the wicked ...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "7  A FIVE STAR BOOK: I just finished reading Whis...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "8  Whispers of the Wicked Saints: This was a easy...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "9  The Worst!: A complete waste of time. Typograp...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "\n",
       "  Unnamed: 2  \n",
       "0        NaN  \n",
       "1        NaN  \n",
       "2        NaN  \n",
       "3        NaN  \n",
       "4        NaN  \n",
       "5        NaN  \n",
       "6        NaN  \n",
       "7        NaN  \n",
       "8        NaN  \n",
       "9        NaN  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8e08a0d6-bcb3-4f39-bd50-f072e34db90e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(700, 3)\n"
     ]
    }
   ],
   "source": [
    "df = df.head(700)\n",
    "print (df.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "bb791531-b689-4e2c-8767-ec6a794fbba6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Whispers of the Wicked Saints: This was a easy to read book that made me want to keep reading on and on, not easy to put down.It left me wanting to read the follow on, which I hope is coming soon. I used to read a lot but have gotten away from it. This book made me want to read again. Very enjoyable.\n"
     ]
    }
   ],
   "source": [
    "example = df['Stuning even for the non-gamer: This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^'][8]\n",
    "print(example)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "fa979c67-25bc-440f-8b70-91a8f4350e0d",
   "metadata": {},
   "outputs": [],
   "source": [
    "tokens = nltk.tokenize.word_tokenize(example)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f92163ec-43f1-4a72-84db-bc688094a8f6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Whispers', 'of', 'the', 'Wicked', 'Saints', ':', 'This', 'was', 'a', 'easy', 'to', 'read', 'book', 'that', 'made', 'me', 'want', 'to', 'keep', 'reading', 'on', 'and', 'on', ',', 'not', 'easy', 'to', 'put', 'down.It', 'left', 'me', 'wanting', 'to', 'read', 'the', 'follow', 'on', ',', 'which', 'I', 'hope', 'is', 'coming', 'soon', '.', 'I', 'used', 'to', 'read', 'a', 'lot', 'but', 'have', 'gotten', 'away', 'from', 'it', '.', 'This', 'book', 'made', 'me', 'want', 'to', 'read', 'again', '.', 'Very', 'enjoyable', '.']\n"
     ]
    }
   ],
   "source": [
    "print(tokens)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2b259c7e-9a2a-4b47-ad80-6cd05ddf54d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "tagged = nltk.pos_tag(tokens)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "65895ee5-83d6-4695-8f83-83c04b3023fa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[('Whispers', 'NNS'), ('of', 'IN'), ('the', 'DT'), ('Wicked', 'JJ'), ('Saints', 'NNS'), (':', ':'), ('This', 'DT'), ('was', 'VBD'), ('a', 'DT'), ('easy', 'JJ'), ('to', 'TO'), ('read', 'VB'), ('book', 'NN'), ('that', 'IN'), ('made', 'VBD'), ('me', 'PRP'), ('want', 'VB'), ('to', 'TO'), ('keep', 'VB'), ('reading', 'NN'), ('on', 'IN'), ('and', 'CC'), ('on', 'IN'), (',', ','), ('not', 'RB'), ('easy', 'JJ'), ('to', 'TO'), ('put', 'VB'), ('down.It', 'NN'), ('left', 'VBD'), ('me', 'PRP'), ('wanting', 'VBG'), ('to', 'TO'), ('read', 'VB'), ('the', 'DT'), ('follow', 'NN'), ('on', 'IN'), (',', ','), ('which', 'WDT'), ('I', 'PRP'), ('hope', 'VBP'), ('is', 'VBZ'), ('coming', 'VBG'), ('soon', 'RB'), ('.', '.'), ('I', 'PRP'), ('used', 'VBD'), ('to', 'TO'), ('read', 'VB'), ('a', 'DT'), ('lot', 'NN'), ('but', 'CC'), ('have', 'VBP'), ('gotten', 'VBN'), ('away', 'RP'), ('from', 'IN'), ('it', 'PRP'), ('.', '.'), ('This', 'DT'), ('book', 'NN'), ('made', 'VBD'), ('me', 'PRP'), ('want', 'VB'), ('to', 'TO'), ('read', 'VB'), ('again', 'RB'), ('.', '.'), ('Very', 'RB'), ('enjoyable', 'JJ'), ('.', '.')]\n"
     ]
    }
   ],
   "source": [
    "print(tagged)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "5652f42a-6af4-4615-bb95-b184b5b8d776",
   "metadata": {},
   "outputs": [],
   "source": [
    "entities =nltk.chunk.ne_chunk(tagged)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e0c9ade1-abb2-4486-bfc7-18c671648bdc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(S\n",
      "  Whispers/NNS\n",
      "  of/IN\n",
      "  the/DT\n",
      "  (ORGANIZATION Wicked/JJ)\n",
      "  Saints/NNS\n",
      "  :/:\n",
      "  This/DT\n",
      "  was/VBD\n",
      "  a/DT\n",
      "  easy/JJ\n",
      "  to/TO\n",
      "  read/VB\n",
      "  book/NN\n",
      "  that/IN\n",
      "  made/VBD\n",
      "  me/PRP\n",
      "  want/VB\n",
      "  to/TO\n",
      "  keep/VB\n",
      "  reading/NN\n",
      "  on/IN\n",
      "  and/CC\n",
      "  on/IN\n",
      "  ,/,\n",
      "  not/RB\n",
      "  easy/JJ\n",
      "  to/TO\n",
      "  put/VB\n",
      "  down.It/NN\n",
      "  left/VBD\n",
      "  me/PRP\n",
      "  wanting/VBG\n",
      "  to/TO\n",
      "  read/VB\n",
      "  the/DT\n",
      "  follow/NN\n",
      "  on/IN\n",
      "  ,/,\n",
      "  which/WDT\n",
      "  I/PRP\n",
      "  hope/VBP\n",
      "  is/VBZ\n",
      "  coming/VBG\n",
      "  soon/RB\n",
      "  ./.\n",
      "  I/PRP\n",
      "  used/VBD\n",
      "  to/TO\n",
      "  read/VB\n",
      "  a/DT\n",
      "  lot/NN\n",
      "  but/CC\n",
      "  have/VBP\n",
      "  gotten/VBN\n",
      "  away/RP\n",
      "  from/IN\n",
      "  it/PRP\n",
      "  ./.\n",
      "  This/DT\n",
      "  book/NN\n",
      "  made/VBD\n",
      "  me/PRP\n",
      "  want/VB\n",
      "  to/TO\n",
      "  read/VB\n",
      "  again/RB\n",
      "  ./.\n",
      "  Very/RB\n",
      "  enjoyable/JJ\n",
      "  ./.)\n"
     ]
    }
   ],
   "source": [
    "print(entities)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "9ba03b09-4635-4f57-9ae6-104234ad564d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.sentiment import SentimentIntensityAnalyzer\n",
    "sai = SentimentIntensityAnalyzer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "a62c55a6-c416-401a-9d2e-883d1bfc6c73",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'neg': 0.033, 'neu': 0.777, 'pos': 0.19, 'compound': 0.8196}"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sai.polarity_scores(example)\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "9a87187e-3ba6-4f5a-81c1-86204fdc8a9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tqdm.notebook import tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "aeaae359-dab4-4df1-bec8-3865d993b9d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "Text = ('Stuning even for the non-gamer: This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "487c8e82-3599-4ba1-8224-8459cfc2a40e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2f3372e1194d4810bae3ea3408c3af10",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/700 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "res = {}\n",
    "for i, row in tqdm(df.iterrows(), total=len(df)):\n",
    "\n",
    "    text = row['Stuning even for the non-gamer: This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^']\n",
    "    myid = row['__label__2']\n",
    "    res[myid] = sai.polarity_scores(text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "e775c587-27b0-4e31-a1d8-8bdec01f63ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "index = ('Stuning even for the non-gamer: This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "15241ab8-e1b8-47a8-ac86-48f27fd8c01a",
   "metadata": {},
   "outputs": [],
   "source": [
    "vaders = pd.DataFrame(res).T\n",
    "vaders = vaders.reset_index().rename(columns={'index' : '__label__2'})\n",
    "vaders = vaders.merge(df, how = 'left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "df4f2048-19e7-4394-af84-cd5c60fe024e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>__label__2</th>\n",
       "      <th>neg</th>\n",
       "      <th>neu</th>\n",
       "      <th>pos</th>\n",
       "      <th>compound</th>\n",
       "      <th>Stuning even for the non-gamer: This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^</th>\n",
       "      <th>Unnamed: 2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.861</td>\n",
       "      <td>0.108</td>\n",
       "      <td>0.7906</td>\n",
       "      <td>The best soundtrack ever to anything.: I'm rea...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.861</td>\n",
       "      <td>0.108</td>\n",
       "      <td>0.7906</td>\n",
       "      <td>Amazing!: This soundtrack is my favorite music...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.861</td>\n",
       "      <td>0.108</td>\n",
       "      <td>0.7906</td>\n",
       "      <td>Excellent Soundtrack: I truly like this soundt...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.861</td>\n",
       "      <td>0.108</td>\n",
       "      <td>0.7906</td>\n",
       "      <td>Remember, Pull Your Jaw Off The Floor After He...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.861</td>\n",
       "      <td>0.108</td>\n",
       "      <td>0.7906</td>\n",
       "      <td>an absolute masterpiece: I am quite sure any o...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>695</th>\n",
       "      <td>__label__1</td>\n",
       "      <td>0.055</td>\n",
       "      <td>0.794</td>\n",
       "      <td>0.151</td>\n",
       "      <td>0.9506</td>\n",
       "      <td>Sophomoric at Best: Having read only one other...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>696</th>\n",
       "      <td>__label__1</td>\n",
       "      <td>0.055</td>\n",
       "      <td>0.794</td>\n",
       "      <td>0.151</td>\n",
       "      <td>0.9506</td>\n",
       "      <td>A Waste of Time: This book is not well written...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>697</th>\n",
       "      <td>__label__1</td>\n",
       "      <td>0.055</td>\n",
       "      <td>0.794</td>\n",
       "      <td>0.151</td>\n",
       "      <td>0.9506</td>\n",
       "      <td>Good story- Poor writing: Very simplistic writ...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>698</th>\n",
       "      <td>__label__1</td>\n",
       "      <td>0.055</td>\n",
       "      <td>0.794</td>\n",
       "      <td>0.151</td>\n",
       "      <td>0.9506</td>\n",
       "      <td>Do I have to give it 1-star?: I've never writt...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>699</th>\n",
       "      <td>__label__1</td>\n",
       "      <td>0.055</td>\n",
       "      <td>0.794</td>\n",
       "      <td>0.151</td>\n",
       "      <td>0.9506</td>\n",
       "      <td>Disappointment on every page: I'm so glad that...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>700 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     __label__2    neg    neu    pos  compound  \\\n",
       "0    __label__2  0.030  0.861  0.108    0.7906   \n",
       "1    __label__2  0.030  0.861  0.108    0.7906   \n",
       "2    __label__2  0.030  0.861  0.108    0.7906   \n",
       "3    __label__2  0.030  0.861  0.108    0.7906   \n",
       "4    __label__2  0.030  0.861  0.108    0.7906   \n",
       "..          ...    ...    ...    ...       ...   \n",
       "695  __label__1  0.055  0.794  0.151    0.9506   \n",
       "696  __label__1  0.055  0.794  0.151    0.9506   \n",
       "697  __label__1  0.055  0.794  0.151    0.9506   \n",
       "698  __label__1  0.055  0.794  0.151    0.9506   \n",
       "699  __label__1  0.055  0.794  0.151    0.9506   \n",
       "\n",
       "    Stuning even for the non-gamer: This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^  \\\n",
       "0    The best soundtrack ever to anything.: I'm rea...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "1    Amazing!: This soundtrack is my favorite music...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "2    Excellent Soundtrack: I truly like this soundt...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "3    Remember, Pull Your Jaw Off The Floor After He...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "4    an absolute masterpiece: I am quite sure any o...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "..                                                 ...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "695  Sophomoric at Best: Having read only one other...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "696  A Waste of Time: This book is not well written...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "697  Good story- Poor writing: Very simplistic writ...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "698  Do I have to give it 1-star?: I've never writt...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "699  Disappointment on every page: I'm so glad that...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "\n",
       "    Unnamed: 2  \n",
       "0          NaN  \n",
       "1          NaN  \n",
       "2          NaN  \n",
       "3          NaN  \n",
       "4          NaN  \n",
       "..         ...  \n",
       "695        NaN  \n",
       "696        NaN  \n",
       "697        NaN  \n",
       "698        NaN  \n",
       "699        NaN  \n",
       "\n",
       "[700 rows x 7 columns]"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vaders"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "598c97ff-6602-46a8-ad43-01eabd754ee2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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AMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwJSrMTsVFBRo7dq1crvdio+P16RJk5SUlFTvtu+88462bt2qb775RpKUmJiocePGnXV7AABwYQn6ysi2bdu0bNkyZWZmKjs7W/Hx8Zo1a5aOHj1a7/aff/65+vfvrz/96U968skndckll+jJJ59UZWVlkwcPAABav6BjJD8/X2lpaUpNTVVcXJyysrIUGhqqTZs21bv9lClTNHz4cCUkJKh79+6aPHmyfD6fdu3a1eTBAwCA1i+oaRqv16vi4mKNHDnSv87pdColJUVFRUUNeoza2lp5vV517NjxrNt4PB55PB7/ssPhUFhYmP/3AFouzlGg7Wqu8zuoGKmqqlJdXZ0iIyMD1kdGRqq0tLRBj/Hqq68qKipKKSkpZ90mLy9Pubm5/uWePXsqOztbXbt2DWa4AAzExMRYDwFAM2mu87tRN7A21po1a/T+++9r5syZCg0NPet2o0aNUkZGhn/5dImVl5fL6/U2+zgBNF5ZWZn1EAA0k2DPb5fL1aALCUHFSEREhJxOp9xud8B6t9t9xtWS//XGG29ozZo1euyxxxQfH/+j24aEhCgkJKTer/l8vmCGDOA84xwF2q7mOr+DuoHV5XIpMTFRhYWF/nV1dXUqLCxUcnLyWff7xz/+odWrV2vGjBm67LLLGj9aAADQ5gT90zQZGRnasGGDNm/erIMHD2rhwoWqra3V4MGD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/7g+//xzLV++XF9//bU6duyoQYMG6Y477lC7du0kSQ888IBGjBihW2+91b/P1KlTdf311/v//Rg7dqzuv/9+7dixQ5999pmioqI0YcIEXXfddf59duzYoaVLl6qiokLJyckaNGhQs/+ZNwZXRgxs2bJF7du315///GfdddddWr16tXbu3ClJeuaZZ3T06FHNmDFDc+bMUc+ePfXEE0/o2LFjkqR3331Xr7/+uu68807NmTNHXbp00dtvv235cvATmnK80Tpt2bJF7dq10+zZs3X33Xdr3bp12rBhgyTpxRdf1Jdffqlp06bpySeflM/n0+zZs+X1eiVJixYtktfr1eOPP66nn35ad955p/+bI4K3YsUKrVmzRmPGjNEzzzyjhx56SJ06dVJNTY1mzZqlDh06aPbs2Xr44Ye1a9cuLVq0KGD/3bt367vvvtPjjz+uCRMmKCcnR3PmzFGHDh305z//WUOHDtXLL7+sb7/9NmC/V155RRkZGcrOzlavXr2UnZ2t77//XpJUWVmp2bNn67LLLtPcuXN13333aePGjVq9enXQry83N1c33XSTnn76aV1zzTV6/vnn/f9+VFRU6C9/+Yv69u2ruXPnasiQIXr11Vcb+SfZvIgRA/Hx8brtttsUExOjQYMGKTExUbt27dKePXu0b98+Pfzww7rssssUExOjCRMmKDw8XP/6178kSQUFBRoyZIhSU1MVGxurzMxM3s22hWvK8UbrdMkll2jixImKjY3VwIEDlZ6ernXr1qmsrEwfffSRJk+erD59+ighIUFTpkxRZWWltm/fLuk/30B69+6tHj166NJLL1Xfvn0D/keNhjtx4oTWr1+vu+66S4MHD1Z0dLQuv/xypaWl6b333tPJkyf14IMPqkePHrryyis1adIkbd26NeDDYDt27Kh77rlHsbGxGjJkiGJjY3Xy5EmNHj1aMTExGjVqlFwul/bs2RPw3MOHD9cvfvELxcXFKSsrS+Hh4dq4caMk6a233tIll1yie++9V927d9cNN9ygsWPHKj8/X3V1dUG9xkGDBmnAgAGKjo7WuHHjVFNTo3379kmS3n77bV166aWaMGGC/+/i6Y9uaWmYpjHwv/HQuXNnHT16VPv371dNTY0mTZoU8PWTJ0/q0KFDkqTS0lINGzYs4OtJSUkBH16IlqUpxxutU69evQI+FiM5OVn5+fk6ePCg2rVrF/BBoRdffLFiY2NVUlIiSbrlllu0cOFC7dy5UykpKbrxxhsVHx9/3l9DW1BSUiKPx6OUlJR6v5aQkBBw1enyyy+Xz+dTaWmp/yNO4uLi5HT+9//tnTp18k/DSZLT6dTFF198xqfT//DDY9u1a6fExET/MS4pKVFycvIZH51SU1OjyspKdenSpcGv8Yd/Ny666CKFhYX5x1JSUqKkpKSzjqslIUYMuFxn/rH7fD7V1NSoc+fO9d4LEB4efh5GhubQlOP9w3+sTjt9OR9tU1pamq666irt2LFDO3fuVF5eniZMmKBbbrnFemitTmhoaJMf4/Q9HKc5HI4zzmmHw6Fz/ckq9T3mqVOnGjS+1vgpL0zTtCCJiYlyu91yOp2Kjo4O+BURESFJio2N1Zdffhmw3/8uo3VoyPGOiIjQd999F7Df119/bTFcBOH0ZfLTvvjiC0VHRysuLk6nTp3SF1984f/a999/r9LSUsXFxfnXdenSRcOGDdMf/vAH/epXv/Lfb4LgREdHKzQ0VLt27Trja927d/dfnTxtz549cjgcio2NbfJz//AYnzp1SsXFxerevbv/uYuKigKiYe/evQoLC1NUVJSk/5z7P5wuqq6u1pEjR4IaQ/fu3c/4/vDDcbUkxEgLkpKSouTkZM2dO1efffaZjhw5or1792rlypX+v1Dp6enauHGjNm/erLKyMq1evVpff/11vf+DRsvWkON95ZVXqri4WFu2bFFZWZlycnJ04MAB45Hjp1RUVGjp0qUqLS3Ve++9p/Xr12vEiBGKiYnRddddp5deekl79uzR/v379cILLygqKsr/ExBLlizRp59+qiNHjqi4uFi7d+/2fxNDcEJDQ/Wb3/xGr7zyirZs2aJDhw6pqKhIGzdu1MCBAxUaGqr58+frwIEDKiws1OLFi3XzzTf7p2ia4q233tKHH36okpISLVq0SMePH/f/xM3w4cP17bff6u9//7tKSkq0fft25eTk6NZbb/VPCV155ZXaunWr/v3vf+vAgQOaP39+wHRRQwwbNkxlZWVavny5/+/i5s2bm/zamgPTNC2Iw+HQ9OnTtXLlSr344ouqqqpSZGSk+vTpo06dOkmSBg4cqMOHD2v58uXyeDy66aabNHjw4DP+J4aWryHH++qrr9aYMWP0yiuvyOPxKDU1VYMGDSJIWribb75ZJ0+e1PTp0+V0OjVixAj98pe/lCT97ne/05IlSzRnzhx5vV716dNH06dP91/6r6ur06JFi1RZWamwsDBdffXVmjhxouXLadXGjBmjdu3aKScnR5WVlercubOGDh2q9u3b69FHH9XixYs1ffr0gB/tPRfGjx+vNWvWaP/+/YqOjta0adP8VzyjoqI0ffp0LV++XFOnTlXHjh01ZMgQjRkzxr//yJEjdeTIEc2ZM0fh4eG6/fbbg74y0qVLFz3yyCNaunSpCgoKlJSUpHHjxulvf/vbOXmN55LD1xonlxDgiSeeUGRkpH7/+99bDwW44M2cOVMJCQm6++67rYcCA6ffZ+Spp54KeH8o/DimaVqZ2tpa5efn65tvvlFJSYlycnK0a9euFvtGNgAA/BSmaVoZh8OhTz75RK+//ro8Ho9iY2P1yCOP6Oc//7n10AAAaBSmaQAAgCmmaQAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAqf8PI9u5ggxOXZQAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(data = vaders)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "d72eac7e-b1a4-4912-b86e-4a2aa7a338a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Transformers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "704057ea-9bfe-45e1-bbcc-71ec1b8b4b99",
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoTokenizer\n",
    "from transformers import AutoModelForSequenceClassification\n",
    "from scipy.special import softmax"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "0be24489-086e-4650-99b9-b7cfb7ae5120",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "MODEL = f\"cardiffnlp/twitter-roberta-base-sentiment\"\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL) \n",
    "model = AutoModelForSequenceClassification.from_pretrained(MODEL)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "94a48bc3-d670-4614-b10f-c1875ecc0654",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Whispers of the Wicked Saints: This was a easy to read book that made me want to keep reading on and on, not easy to put down.It left me wanting to read the follow on, which I hope is coming soon. I used to read a lot but have gotten away from it. This book made me want to read again. Very enjoyable.\n"
     ]
    }
   ],
   "source": [
    "print(example)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "0f7c7705-017c-44a7-8443-f14bea9f4e9d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def polarity_scores_roberta(example):\n",
    "\n",
    "    encoded_text = tokenizer(example, return_tensors= 'pt')\n",
    "    output = model(**encoded_text)\n",
    "    scores = output[0][0].detach().numpy()\n",
    "    scores = softmax(scores)\n",
    "    scores_dict = {'roberta_neg' : scores[0],\n",
    "                   'roberta_neu' : scores[1],\n",
    "                   'roberta_pos' : scores[2]}\n",
    "    return scores_dict\n",
    "              \n",
    "               "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "da682326-a544-4956-a36b-e9d496276bcb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8c6e066221014d439724ec8e8c7bc5fe",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/700 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "res = {}\n",
    "for i, row in tqdm(df.iterrows(), total=len(df)):\n",
    "\n",
    "    text = row['Stuning even for the non-gamer: This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^']\n",
    "    myid = row['__label__2']\n",
    "   \n",
    "    \n",
    "    roberta_result = polarity_scores_roberta(text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "336b7e3a-458e-4d9c-b280-50e80b3abd75",
   "metadata": {},
   "outputs": [],
   "source": [
    "roberta = pd.DataFrame(res).T\n",
    "roberta = vaders.reset_index().rename(columns={'index' : '__label__2'})\n",
    "roberta = vaders.merge(df, how = 'left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "75e1371e-a337-4055-9ffc-9123489e83b6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>__label__2</th>\n",
       "      <th>neg</th>\n",
       "      <th>neu</th>\n",
       "      <th>pos</th>\n",
       "      <th>compound</th>\n",
       "      <th>Stuning even for the non-gamer: This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^</th>\n",
       "      <th>Unnamed: 2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.861</td>\n",
       "      <td>0.108</td>\n",
       "      <td>0.7906</td>\n",
       "      <td>The best soundtrack ever to anything.: I'm rea...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.861</td>\n",
       "      <td>0.108</td>\n",
       "      <td>0.7906</td>\n",
       "      <td>Amazing!: This soundtrack is my favorite music...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.861</td>\n",
       "      <td>0.108</td>\n",
       "      <td>0.7906</td>\n",
       "      <td>Excellent Soundtrack: I truly like this soundt...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.861</td>\n",
       "      <td>0.108</td>\n",
       "      <td>0.7906</td>\n",
       "      <td>Remember, Pull Your Jaw Off The Floor After He...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>__label__2</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.861</td>\n",
       "      <td>0.108</td>\n",
       "      <td>0.7906</td>\n",
       "      <td>an absolute masterpiece: I am quite sure any o...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>695</th>\n",
       "      <td>__label__1</td>\n",
       "      <td>0.055</td>\n",
       "      <td>0.794</td>\n",
       "      <td>0.151</td>\n",
       "      <td>0.9506</td>\n",
       "      <td>Sophomoric at Best: Having read only one other...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>696</th>\n",
       "      <td>__label__1</td>\n",
       "      <td>0.055</td>\n",
       "      <td>0.794</td>\n",
       "      <td>0.151</td>\n",
       "      <td>0.9506</td>\n",
       "      <td>A Waste of Time: This book is not well written...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>697</th>\n",
       "      <td>__label__1</td>\n",
       "      <td>0.055</td>\n",
       "      <td>0.794</td>\n",
       "      <td>0.151</td>\n",
       "      <td>0.9506</td>\n",
       "      <td>Good story- Poor writing: Very simplistic writ...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>698</th>\n",
       "      <td>__label__1</td>\n",
       "      <td>0.055</td>\n",
       "      <td>0.794</td>\n",
       "      <td>0.151</td>\n",
       "      <td>0.9506</td>\n",
       "      <td>Do I have to give it 1-star?: I've never writt...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>699</th>\n",
       "      <td>__label__1</td>\n",
       "      <td>0.055</td>\n",
       "      <td>0.794</td>\n",
       "      <td>0.151</td>\n",
       "      <td>0.9506</td>\n",
       "      <td>Disappointment on every page: I'm so glad that...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>700 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     __label__2    neg    neu    pos  compound  \\\n",
       "0    __label__2  0.030  0.861  0.108    0.7906   \n",
       "1    __label__2  0.030  0.861  0.108    0.7906   \n",
       "2    __label__2  0.030  0.861  0.108    0.7906   \n",
       "3    __label__2  0.030  0.861  0.108    0.7906   \n",
       "4    __label__2  0.030  0.861  0.108    0.7906   \n",
       "..          ...    ...    ...    ...       ...   \n",
       "695  __label__1  0.055  0.794  0.151    0.9506   \n",
       "696  __label__1  0.055  0.794  0.151    0.9506   \n",
       "697  __label__1  0.055  0.794  0.151    0.9506   \n",
       "698  __label__1  0.055  0.794  0.151    0.9506   \n",
       "699  __label__1  0.055  0.794  0.151    0.9506   \n",
       "\n",
       "    Stuning even for the non-gamer: This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^  \\\n",
       "0    The best soundtrack ever to anything.: I'm rea...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "1    Amazing!: This soundtrack is my favorite music...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "2    Excellent Soundtrack: I truly like this soundt...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "3    Remember, Pull Your Jaw Off The Floor After He...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "4    an absolute masterpiece: I am quite sure any o...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "..                                                 ...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "695  Sophomoric at Best: Having read only one other...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "696  A Waste of Time: This book is not well written...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "697  Good story- Poor writing: Very simplistic writ...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "698  Do I have to give it 1-star?: I've never writt...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "699  Disappointment on every page: I'm so glad that...                                                                                                                                                                                                                                                                                                                                                                                           \n",
       "\n",
       "    Unnamed: 2  \n",
       "0          NaN  \n",
       "1          NaN  \n",
       "2          NaN  \n",
       "3          NaN  \n",
       "4          NaN  \n",
       "..         ...  \n",
       "695        NaN  \n",
       "696        NaN  \n",
       "697        NaN  \n",
       "698        NaN  \n",
       "699        NaN  \n",
       "\n",
       "[700 rows x 7 columns]"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "roberta\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "d117bfec-1aec-4dcd-82ad-954e3988d1fb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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AMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwJSrMTsVFBRo7dq1crvdio+P16RJk5SUlFTvtu+88462bt2qb775RpKUmJiocePGnXV7AABwYQn6ysi2bdu0bNkyZWZmKjs7W/Hx8Zo1a5aOHj1a7/aff/65+vfvrz/96U968skndckll+jJJ59UZWVlkwcPAABav6BjJD8/X2lpaUpNTVVcXJyysrIUGhqqTZs21bv9lClTNHz4cCUkJKh79+6aPHmyfD6fdu3a1eTBAwCA1i+oaRqv16vi4mKNHDnSv87pdColJUVFRUUNeoza2lp5vV517NjxrNt4PB55PB7/ssPhUFhYmP/3AFouzlGg7Wqu8zuoGKmqqlJdXZ0iIyMD1kdGRqq0tLRBj/Hqq68qKipKKSkpZ90mLy9Pubm5/uWePXsqOztbXbt2DWa4AAzExMRYDwFAM2mu87tRN7A21po1a/T+++9r5syZCg0NPet2o0aNUkZGhn/5dImVl5fL6/U2+zgBNF5ZWZn1EAA0k2DPb5fL1aALCUHFSEREhJxOp9xud8B6t9t9xtWS//XGG29ozZo1euyxxxQfH/+j24aEhCgkJKTer/l8vmCGDOA84xwF2q7mOr+DuoHV5XIpMTFRhYWF/nV1dXUqLCxUcnLyWff7xz/+odWrV2vGjBm67LLLGj9aAADQ5gT90zQZGRnasGGDNm/erIMHD2rhwoWqra3V4MGD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/7g+//xzLV++XF9//bU6duyoQYMG6Y477lC7du0kSQ888IBGjBihW2+91b/P1KlTdf311/v//Rg7dqzuv/9+7dixQ5999pmioqI0YcIEXXfddf59duzYoaVLl6qiokLJyckaNGhQs/+ZNwZXRgxs2bJF7du315///GfdddddWr16tXbu3ClJeuaZZ3T06FHNmDFDc+bMUc+ePfXEE0/o2LFjkqR3331Xr7/+uu68807NmTNHXbp00dtvv235cvATmnK80Tpt2bJF7dq10+zZs3X33Xdr3bp12rBhgyTpxRdf1Jdffqlp06bpySeflM/n0+zZs+X1eiVJixYtktfr1eOPP66nn35ad955p/+bI4K3YsUKrVmzRmPGjNEzzzyjhx56SJ06dVJNTY1mzZqlDh06aPbs2Xr44Ye1a9cuLVq0KGD/3bt367vvvtPjjz+uCRMmKCcnR3PmzFGHDh305z//WUOHDtXLL7+sb7/9NmC/V155RRkZGcrOzlavXr2UnZ2t77//XpJUWVmp2bNn67LLLtPcuXN13333aePGjVq9enXQry83N1c33XSTnn76aV1zzTV6/vnn/f9+VFRU6C9/+Yv69u2ruXPnasiQIXr11Vcb+SfZvIgRA/Hx8brtttsUExOjQYMGKTExUbt27dKePXu0b98+Pfzww7rssssUExOjCRMmKDw8XP/6178kSQUFBRoyZIhSU1MVGxurzMxM3s22hWvK8UbrdMkll2jixImKjY3VwIEDlZ6ernXr1qmsrEwfffSRJk+erD59+ighIUFTpkxRZWWltm/fLuk/30B69+6tHj166NJLL1Xfvn0D/keNhjtx4oTWr1+vu+66S4MHD1Z0dLQuv/xypaWl6b333tPJkyf14IMPqkePHrryyis1adIkbd26NeDDYDt27Kh77rlHsbGxGjJkiGJjY3Xy5EmNHj1aMTExGjVqlFwul/bs2RPw3MOHD9cvfvELxcXFKSsrS+Hh4dq4caMk6a233tIll1yie++9V927d9cNN9ygsWPHKj8/X3V1dUG9xkGDBmnAgAGKjo7WuHHjVFNTo3379kmS3n77bV166aWaMGGC/+/i6Y9uaWmYpjHwv/HQuXNnHT16VPv371dNTY0mTZoU8PWTJ0/q0KFDkqTS0lINGzYs4OtJSUkBH16IlqUpxxutU69evQI+FiM5OVn5+fk6ePCg2rVrF/BBoRdffLFiY2NVUlIiSbrlllu0cOFC7dy5UykpKbrxxhsVHx9/3l9DW1BSUiKPx6OUlJR6v5aQkBBw1enyyy+Xz+dTaWmp/yNO4uLi5HT+9//tnTp18k/DSZLT6dTFF198xqfT//DDY9u1a6fExET/MS4pKVFycvIZH51SU1OjyspKdenSpcGv8Yd/Ny666CKFhYX5x1JSUqKkpKSzjqslIUYMuFxn/rH7fD7V1NSoc+fO9d4LEB4efh5GhubQlOP9w3+sTjt9OR9tU1pamq666irt2LFDO3fuVF5eniZMmKBbbrnFemitTmhoaJMf4/Q9HKc5HI4zzmmHw6Fz/ckq9T3mqVOnGjS+1vgpL0zTtCCJiYlyu91yOp2Kjo4O+BURESFJio2N1Zdffhmw3/8uo3VoyPGOiIjQd999F7Df119/bTFcBOH0ZfLTvvjiC0VHRysuLk6nTp3SF1984f/a999/r9LSUsXFxfnXdenSRcOGDdMf/vAH/epXv/Lfb4LgREdHKzQ0VLt27Trja927d/dfnTxtz549cjgcio2NbfJz//AYnzp1SsXFxerevbv/uYuKigKiYe/evQoLC1NUVJSk/5z7P5wuqq6u1pEjR4IaQ/fu3c/4/vDDcbUkxEgLkpKSouTkZM2dO1efffaZjhw5or1792rlypX+v1Dp6enauHGjNm/erLKyMq1evVpff/11vf+DRsvWkON95ZVXqri4WFu2bFFZWZlycnJ04MAB45Hjp1RUVGjp0qUqLS3Ve++9p/Xr12vEiBGKiYnRddddp5deekl79uzR/v379cILLygqKsr/ExBLlizRp59+qiNHjqi4uFi7d+/2fxNDcEJDQ/Wb3/xGr7zyirZs2aJDhw6pqKhIGzdu1MCBAxUaGqr58+frwIEDKiws1OLFi3XzzTf7p2ia4q233tKHH36okpISLVq0SMePH/f/xM3w4cP17bff6u9//7tKSkq0fft25eTk6NZbb/VPCV155ZXaunWr/v3vf+vAgQOaP39+wHRRQwwbNkxlZWVavny5/+/i5s2bm/zamgPTNC2Iw+HQ9OnTtXLlSr344ouqqqpSZGSk+vTpo06dOkmSBg4cqMOHD2v58uXyeDy66aabNHjw4DP+J4aWryHH++qrr9aYMWP0yiuvyOPxKDU1VYMGDSJIWribb75ZJ0+e1PTp0+V0OjVixAj98pe/lCT97ne/05IlSzRnzhx5vV716dNH06dP91/6r6ur06JFi1RZWamwsDBdffXVmjhxouXLadXGjBmjdu3aKScnR5WVlercubOGDh2q9u3b69FHH9XixYs1ffr0gB/tPRfGjx+vNWvWaP/+/YqOjta0adP8VzyjoqI0ffp0LV++XFOnTlXHjh01ZMgQjRkzxr//yJEjdeTIEc2ZM0fh4eG6/fbbg74y0qVLFz3yyCNaunSpCgoKlJSUpHHjxulvf/vbOXmN55LD1xonlxDgiSeeUGRkpH7/+99bDwW44M2cOVMJCQm6++67rYcCA6ffZ+Spp54KeH8o/DimaVqZ2tpa5efn65tvvlFJSYlycnK0a9euFvtGNgAA/BSmaVoZh8OhTz75RK+//ro8Ho9iY2P1yCOP6Oc//7n10AAAaBSmaQAAgCmmaQAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAqf8PI9u5ggxOXZQAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(data = roberta )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5f58c25c-65a7-4d82-bfa7-a45c8572c9c3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
