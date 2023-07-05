import torch
import numpy as np
from src.models.EA_LSTM.model import weightedLSTM
from src.datasets.dataloader import MyDataset, create_vocab


def test(args):
    vocab, poetrys = create_vocab(args.data)
    # 词汇表长度
    args.vocab_size = len(vocab)
    int2char = np.array(vocab)
    valid_dataset = MyDataset(vocab, poetrys, args, train=False)

    model = weightedLSTM(6110, 256, 128, 2, [1.0] * 80, False)
    model.load_state_dict(torch.load(args.save_path))

    input_example_batch, target_example_batch = valid_dataset[0]
    example_batch_predictions = model(input_example_batch)
    predicted_id = torch.distributions.Categorical(example_batch_predictions).sample()
    predicted_id = torch.squeeze(predicted_id, -1).numpy()
    print("Input: \n", repr("".join(int2char[input_example_batch])))
    print()
    print("Predictions: \n", repr("".join(int2char[predicted_id])))
