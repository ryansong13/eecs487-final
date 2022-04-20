import torch

class RNNModel(torch.nn.Module):
  def __init__(self, input_size, hidden_size, n_layers, device):
    super(RNNModel, self).__init__()

    self.input_size = input_size # NOTE: "input" = number of expected features (should that just be 1 or len(tweets)?)
    self.hidden_size = hidden_size
    self.n_layers = n_layers 
    self.device = device

    self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size)
    self.linear = torch.nn.Linear(hidden_size, 1) # OUTPUT = 1 (binary classification)

  def forward(self, tweet):
    batch_size = tweet.size(0)
    hidden = self.init_hidden(batch_size)
    out, hidden = self.rnn(tweet)
    out = self.linear(out)
    return out, hidden

  def init_hidden(self, batch_size):
    hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)
    return hidden

