def unique_words(sentence):
    unique_sentence = []

    for word in sentence:
        if word not in unique_sentence:
            unique_sentence.append(word)

    return unique_sentence
