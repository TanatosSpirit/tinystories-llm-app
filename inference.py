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
        base_chars = ['\n'] + [chr(i) for i in range(32, 127)]
        self.ascii_code_to_id = {ord(char): i + 3 for i, char in enumerate(base_chars)}

    def generate_text(self, prompt, **gen_params):
        story_start = []

        prompt = prompt.strip()

        if prompt:
            prompt += " "

        yield {'last': False, 'next_token': prompt, 'confidence': -1}

        for ch in prompt:
            code = ord(ch)
            if 32 <= code <= 126 or code == 10:
                story_start.append(self.ascii_code_to_id[code])
            else:
                raise RuntimeError("An unexpected char code.")

        prompt = self.tokenizer.encode(story_start)
        prompt = np.array(prompt[:-1], dtype=np.int64)

        max_seq_length = 512
        decoded_sequence = np.full(
            max_seq_length,
            fill_value=0,
            dtype=np.int64,
        )
        prompt_length = prompt.shape[0]
        decoded_sequence[:prompt_length] = prompt[:prompt_length]

        finished_sequences = np.zeros(1, dtype=bool)

        max_seq_length = max_seq_length - prompt_length

        while (not finished_sequences.all()) and (prompt_length < max_seq_length):
            decoded_sequence_current = decoded_sequence[:prompt_length]
            prediction = self.model(decoded_sequence_current)
            current_output = prediction[:, -1, :]
            if gen_params['method'] == 0: # Greedy
                current_token = np.argmax(current_output, axis=1)
                probs = np.exp(current_output - np.max(current_output, axis=-1, keepdims=True))
                probs = probs / np.sum(probs, axis=-1, keepdims=True)
                confidence = probs[0,current_token]

            elif gen_params['method'] == 1: # Only Temp
                temp = max(float(gen_params['temp']), float(1e-9))
                logits = current_output / temp
                probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs = probs / np.sum(probs, axis=-1, keepdims=True)
                current_token = np.array([np.random.choice(len(p), p=p) for p in probs])
                confidence = probs[0, current_token]

            elif gen_params['method'] == 2: # top-k + temp
                k = gen_params['top_k']
                temp = max(float(gen_params['temp']), float(1e-9))
                logits = current_output[0] / temp

                top_k_idx = np.argpartition(logits, -k)[-k:]
                top_k_logits = logits[top_k_idx]

                probs = np.exp(top_k_logits - np.max(top_k_logits))
                probs = probs / probs.sum()

                current_token = np.random.choice(top_k_idx, p=probs)

                probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs = probs / np.sum(probs, axis=-1, keepdims=True)
                confidence = probs[current_token]

            elif gen_params['method'] == 3: # top-p + temp
                p = gen_params['top_p']
                temp = max(float(gen_params['temp']), float(1e-9))

                logits = current_output[0] / temp

                sorted_idx = np.argsort(logits)[::-1]

                sorted_logits = logits[sorted_idx]
                max_logit = sorted_logits[0]
                probs = np.exp(sorted_logits - max_logit)
                probs /= probs.sum()


                cumprobs = np.cumsum(probs)
                cutoff_idx = np.searchsorted(cumprobs, p, side='right') + 1
                # print(cutoff_idx)
                cutoff_idx = min(cutoff_idx, len(cumprobs))
                # print("max_prob:", probs[0], "p:", p, "cutoff_idx:", cutoff_idx)

                top_p_idx = sorted_idx[:cutoff_idx]
                top_p_probs = probs[:cutoff_idx]
                top_p_probs_cached = top_p_probs

                # Перенормализация (если p < 1.0)
                if cutoff_idx < len(probs):
                    top_p_probs /= top_p_probs.sum()

                # print("Sampling from", len(top_p_idx), "tokens:", top_p_idx[:5])
                current_token = np.random.choice(top_p_idx, p=top_p_probs)
                token_pos = np.where(top_p_idx == current_token)[0][0]
                confidence = top_p_probs_cached[token_pos]


            next_token = np.where(
                finished_sequences,
                1,
                current_token
            )

            decoded_sequence[prompt_length] = next_token

            just_finished = (next_token == 1)
            finished_sequences = np.logical_or(finished_sequences, just_finished)

            # Декодируем только новый токен и возвращаем его
            if not finished_sequences.all():
                new_token_text = self.tokenizer.decode(decoded_sequence[prompt_length:prompt_length + 1])
                yield {'last': False, 'next_token': new_token_text, 'confidence': confidence}

            prompt_length += 1

        yield {'last': True, 'length' : prompt_length}
