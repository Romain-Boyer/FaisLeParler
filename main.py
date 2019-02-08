import io
import tkinter as Tkinter

import numpy as np
import pandas as pd

import settings


class chat_bot(Tkinter.Tk):
    """
    Interface Tkinter
    """

    def __init__(self, parent, script_2_learn, df, word2vec_model):
        Tkinter.Tk.__init__(self, parent)
        self.parent = parent
        self.initialize()
        self.script_2_learn = script_2_learn
        self.word2vec_model = word2vec_model
        self.df = df

    def initialize(self):
        # Gestionnaire de Layout
        self.grid()
        self.geometry("500x300+500+300")
        self.row = 1

        # Ajouter un champ texte
        self.entryVariable = Tkinter.StringVar()
        self.entry = Tkinter.Entry(self, textvariable=self.entryVariable)
        self.entry.grid(column=0, row=0, sticky="EW")
        self.entry.bind("<Return>", self.OnPressEnter)

        # Ajouter le bouton
        button = Tkinter.Button(self, text=u"Send !", command=self.OnButtonClick)
        button.grid(column=1, row=0)

        # Ajouter une zone d'affichage
        self.labelVariable = Tkinter.StringVar()
        self.labelVariable.set(u"Demarrez la conversation !")
        label = Tkinter.Label(
            self,
            textvariable=self.labelVariable,
            anchor="w",
            fg="white",
            bg="OliveDrab2",
        )
        label.grid(column=0, row=self.row, columnspan=2, sticky="EW")

        # Configurer le fond
        self.grid_columnconfigure(0, weight=1)
        self.resizable(True, True)
        self.update()
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)

    def DisplayMessagesSent(self, text):
        """
        Display the message sent by user
        :param text: input text
        """
        label = Tkinter.Label(
            self, textvariable=text, anchor="w", fg="white", bg="royal blue"
        )
        label.grid(column=0, row=self.row, columnspan=2, sticky="EW")
        self.row += 1

    def DisplayMessagesRec(self, text):
        """
        Display the message sent by the bot
        :param text:
        """
        label = Tkinter.Label(
            self, textvariable=text, anchor="w", fg="black", bg="gray86"
        )
        label.grid(column=0, row=self.row, columnspan=2, sticky="EW")
        self.row += 1

    def OnButtonClick(self):
        """
        Save the message sent by user, compute and return the bot's answer
        """
        self.text_input = Tkinter.StringVar()
        self.text_output = Tkinter.StringVar()

        self.text_input.set("Me : " + self.entryVariable.get())
        self.text_output.set(
            "Them : "
            + answer_from_movie(self.entryVariable.get(), self.script_2_learn, self.df)
        )

        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)

        self.DisplayMessagesSent(self.text_input)
        self.DisplayMessagesRec(self.text_output)

    def OnPressEnter(self, event):
        self.OnButtonClick()


def load_script(fpath):
    """
    Load the script in a csv format.
    The columns desired are :
        temps: time of the sentence in the movie
        scene: number of the current scene
        acteur: name of the actor
        phrase: the sentence said
        utilisable: 0 if this sentence is the first sentence of a scene, else 1
    :param fpath: path to the script
    :return: array of the mean sentences of the movie
    """
    # Load the script
    df = pd.read_csv(fpath)

    # Compute the mean value of each sentence and add it to an list
    script_2_learn = []
    for i in df.index:
        if df.loc[i, "utilisable"] == 0:
            script_2_learn.append(np.zeros(300))
        else:
            script_2_learn.append(_mean_sentence(df.loc[i - 1, "phrase"]))
    script_2_learn = np.array(script_2_learn)

    return script_2_learn, df


def load_wordvec(fname, nmax):
    """
    Load the word2vec model
    :param fname: path to the model
    :param nmax: number of words selected in the model (50k is good)
    :return: dict
    """
    word2vec_ = {}
    with io.open(fname, encoding="utf-8") as f:
        deleted_words = 0  # Add a counter to keep {{nmax}} words with deleted words
        next(f)
        for i, line in enumerate(f):
            word, vec = line.split(" ", 1)
            if word not in [
                ",",
                ".",
                "#",
                "!",
                ",",
                '"',
                "'",
                ":",
                ";",
                "(",
                ")",
                "/",
                "</s>",
                "-",
            ]:  # Remove some punctuations
                word2vec_[word] = np.fromstring(vec, sep=" ")
            else:
                deleted_words += 1
            if i == (nmax + deleted_words - 1):
                break
    print("Loaded %s pretrained word vectors" % (len(word2vec_)))
    return word2vec_


def answer_from_movie(sentence, script, df):
    """
    Find the closest sentence in the corpus to the sentence sent by a user.
    And return the sentence after this sentence.
    :param sentence: sentence from user
    :param script: corpus to find answers
    :return: answer from the corpus
    """
    # Compute the value of the sentence with Word2Vec
    sentence = sentence.replace(",", "")
    sent = _mean_sentence(sentence)

    # Compute a similarity score with all sentences in the corpus to find the closest
    score = []
    for s in script:
        score.append(_cosine_sim(sent, s))

    best_index = np.argmax(score)

    return df.loc[best_index, "phrase"].replace("\xa0", "").replace("(Voix off.)", "")


def _mean_sentence(sentence):
    """
    Compute the vector associated to a sentence
    :param sentence: input text
    :return: np.array
    """
    sent = np.array(
        [word2vec[word] for word in sentence.split(" ") if word in word2vec]
    )
    if sent.any():
        return sent.mean(axis=0)
    else:
        return np.zeros(300)


def _cosine_sim(sent1, sent2):
    """
    Compute the cosine similarty between 2 vectors
    :param sent1: word2vec vector 1
    :param sent2: word2vec vector 2
    :return: Return a distance
    """
    if (np.linalg.norm(sent1) == 0) | (np.linalg.norm(sent2) == 0):
        return 0
    else:
        return np.dot(sent1, sent2) / (np.linalg.norm(sent1) * np.linalg.norm(sent2))


if __name__ == "__main__":
    # Load the Word2Vec model
    # Download here : https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fr.vec
    word2vec = load_wordvec(settings.WIKIFR_PATH, nmax=50000)
    script_2_learn, df = load_script(settings.LACLASSEAMERICAINE_PATH)

    app = chat_bot(None, script_2_learn, df, word2vec)
    app.title("LCA")
    app.mainloop()
