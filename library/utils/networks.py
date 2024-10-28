import torch
from torch import nn

class Cfg:
    #Сколько шагов запоминает
    memory_length = 40


class QCNN(nn.Module):
    def __init__(self, in_channels, acs_dim):
        super(QCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(7, 80),
            nn.ReLU(),
        )
        
        # GRU для работы с последовательностью запомненных состояний агента
        self.gru = nn.GRU(input_size=7, hidden_size=80, num_layers=1, batch_first=True)

        self.head = nn.Sequential(
            #nn.Linear(1616+100+48, 612),
            nn.Linear(1616+80, 612),
            nn.ReLU(),
            nn.LayerNorm(612),
            nn.Linear(612, acs_dim),
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor, prev_proprios: torch.Tensor, dop: torch.Tensor) -> torch.Tensor:
        # Image stream
        bs, h, w, c = image.shape
        image = image.permute(0, 3, 1, 2)
        image = image / 255.
        image_repr = self.cnn(image)
        proprio[:, 0] = proprio[:, 0] / 100.  # Нормируем деньги. Их может быть заметно большще 1
        proprio_repr = self.mlp(proprio)
        prev_proprios = prev_proprios.view(prev_proprios.size(0), Cfg.memory_length, 7)  # batch_size, seq_len, input_size
        #prev_proprios_repr = self.memory_mlp(torch.cat((proprio, prev_proprios), dim=1))
        # Передаем последовательность через GRU
        _, hidden_gru = self.gru(prev_proprios)  # hidden_gru имеет размер (num_layers, batch_size, hidden_size)
        # Мы берем последнее скрытое состояние GRU, т.е. hidden[-1] для дальнейшей обработки
        prev_proprios_repr = hidden_gru[-1]  # (batch_size, hidden_size)
        
        hidden = torch.cat([image_repr, proprio_repr, prev_proprios_repr], 1)
        q_vals = self.head(hidden)
        return q_vals
