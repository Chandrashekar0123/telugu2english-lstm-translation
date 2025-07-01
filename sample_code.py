import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# Step 1: Add <start> and <end> tokens to English sentences
telugu_sentences = [
    "నేను పాఠశాలకి వెళ్ళిపోతున్నాను",
    "ఈ రోజు చాలా మంచిది",
    "మీరు ఎలా ఉన్నారు?",
    "నేను మంచి విద్యార్థిని",
    "నేను ఇంగ్లీష్ మాట్లాడగలుగుతాను"
]

english_sentences = [
    "<start> I am going to school <end>",
    "<start> Today is very good <end>",
    "<start> How are you? <end>",
    "<start> I am a good student <end>",
    "<start> I can speak English <end>"
]

# Tokenize Telugu
telugu_tokenizer = Tokenizer()
telugu_tokenizer.fit_on_texts(telugu_sentences)
telugu_vocab_size = len(telugu_tokenizer.word_index) + 1

# Tokenize English
english_tokenizer = Tokenizer(filters='')
english_tokenizer.fit_on_texts(english_sentences)
english_vocab_size = len(english_tokenizer.word_index) + 1

# Sequences
telugu_sequences = telugu_tokenizer.texts_to_sequences(telugu_sentences)
english_sequences = english_tokenizer.texts_to_sequences(english_sentences)

# Padding
max_telugu_length = max(len(seq) for seq in telugu_sequences)
max_english_length = max(len(seq) for seq in english_sequences)

telugu_sequences_padded = pad_sequences(telugu_sequences, maxlen=max_telugu_length, padding='post')
english_sequences_padded = pad_sequences(english_sequences, maxlen=max_english_length, padding='post')

# English input (without last token) and output (without first token)
english_input = english_sequences_padded[:, :-1]
english_output = english_sequences_padded[:, 1:]
english_output = to_categorical(english_output, num_classes=english_vocab_size)

# Encoder
encoder_input = Input(shape=(max_telugu_length,))
encoder_embedding = Embedding(telugu_vocab_size, 256)(encoder_input)
encoder_output, state_h, state_c = LSTM(256, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_input = Input(shape=(max_english_length - 1,))
decoder_embedding = Embedding(english_vocab_size, 256)(decoder_input)
decoder_output, _, _ = LSTM(256, return_sequences=True, return_state=True)(decoder_embedding, initial_state=encoder_states)
decoder_output = Dense(english_vocab_size, activation='softmax')(decoder_output)

# Seq2Seq Model
model = Model([encoder_input, decoder_input], decoder_output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit([telugu_sequences_padded, english_input], english_output, batch_size=2, epochs=300)

# Encoder model for inference
encoder_model = Model(encoder_input, encoder_states)

# Decoder model for inference
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding_inf = decoder_embedding
decoder_lstm_inf = model.layers[5]
decoder_dense_inf = model.layers[6]

decoder_output_inf, state_h_inf, state_c_inf = decoder_lstm_inf(decoder_embedding_inf, initial_state=decoder_states_inputs)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_output_inf = decoder_dense_inf(decoder_output_inf)

decoder_model = Model(
    [decoder_input] + decoder_states_inputs,
    [decoder_output_inf] + decoder_states_inf
)

# Translate function
def predict_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = english_tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = english_tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_english_length:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

# Test it
input_text = "నేను ఇంగ్లీష్ మాట్లాడగలుగుతాను"
input_seq = telugu_tokenizer.texts_to_sequences([input_text])
input_seq = pad_sequences(input_seq, maxlen=max_telugu_length, padding='post')
translation = predict_sequence(input_seq)

print("Input:", input_text)
print("Predicted Translation:", translation)
