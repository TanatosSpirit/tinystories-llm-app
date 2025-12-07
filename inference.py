import onnxruntime as ort
import numpy as np
from tokenizer import BPETokenizer


class ONNXConverter:
    def __init__(self, path_config):
        self.path_config = path_config
        self.session = None

    def load_model(self):
        """Loads converted model."""
        self.session = ort.InferenceSession(self.path_config)

    def __call__(self, inputs) -> np.ndarray:
        if self.session is None:
            raise Exception('Model is not loaded')

        return self.session.run(None, {'inputs': inputs})[0]


class Inference:
    def __init__(self, path_config):
        self.path_config = path_config

        self.__prepare_model()

    def __prepare_model(self):
        self.model = ONNXConverter(self.path_config)
        self.model.load_model()

    def prepare_data(self, data: np.ndarray):
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        return data

    def __call__(self, data: np.ndarray) -> np.ndarray:
        inputs = self.prepare_data(data)
        output = self.model(inputs)
        return output


class Narrator:
    def __init__(self, model_path, tokenizer_path):
        self.model = Inference(model_path)
        self.tokenizer = BPETokenizer(tokenizer_path)

    def generate_text(self, prompt):
        story_start = []

        prompt = prompt.strip()

        if prompt:
            prompt += " "

        for ch in prompt:
            code = ord(ch)
            if 32 <= code <= 126 or ch in '\n':
                story_start.append(code)
            else:
                story_start.append(ord(' '))

        prompt = self.tokenizer.encode(story_start)
        prompt = np.array(prompt[:-1], dtype=np.int64)

        decoded_sequence = np.full(
            (350),
            fill_value=0,
            dtype=np.int64,
        )
        prompt_length = prompt.shape[0]
        decoded_sequence[:prompt_length] = prompt[:prompt_length]

        finished_sequences = np.zeros(1, dtype=bool)

        max_seq_length = 350
        max_seq_length = max_seq_length - prompt_length

        while (not finished_sequences.all()) and (prompt_length < max_seq_length):
            decoded_sequence_current = decoded_sequence[:prompt_length]
            prediction = self.model(decoded_sequence_current)
            current_output = prediction[:, -1, :]
            current_token = np.argmax(current_output, axis=1)

            next_token = np.where(
                finished_sequences,
                1,
                current_token
            )

            decoded_sequence[prompt_length] = next_token

            just_finished = (next_token == 1)
            finished_sequences = np.logical_or(finished_sequences, just_finished)

            prompt_length += 1

        return self.tokenizer.decode(decoded_sequence), prompt_length
