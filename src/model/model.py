import sys
# we would like to be in the src directory to have access to main files
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.distributions.uniform import Uniform
import numpy as np

class im2latex(nn.Module):
    r""" Create the image to latex converter model.

    # Arguments

        encoder_lstm_units:     The dimensionality of the output space for encoder LSTM layers
        decoder_lstm_units:     The dimensionality of the output space for decoder LSTM layers
        vocab_list:             The array of possible outputs of the language model
        embedding_size:         The max length of the equation


    # Example

    .. code:: python
    latex_model = im2latx(encoder_lstm_units, decoder_lstm_units, vocab_list)

    model = latex_model.model
    """

    def __init__(self,
                vocab_size,
                dropout=0.2,
                encoder_lstm_units=256,
                decoder_lstm_units=256,
                embedding_size=64):
        super(im2latex, self).__init__()
        self.name = 'im2latex'
        self.encoder_lstm_units = encoder_lstm_units
        self.decoder_lstm_units = decoder_lstm_units
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        # encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            #nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 64, 3, 1),
            #nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1),
            #nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            # nn.Conv2d(128, 256, 3, 1),
            # #nn.BatchNorm2d(num_features=256),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, self.encoder_lstm_units, 3, 1, 0),
            #nn.BatchNorm2d(num_features=self.encoder_lstm_units),
            nn.ReLU()
        )

        # token_decoder/encoder
        self.rnn_decoder = nn.LSTMCell(self.decoder_lstm_units+self.embedding_size, self.decoder_lstm_units)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        self.init_wh = nn.Linear(self.encoder_lstm_units, self.decoder_lstm_units)
        self.init_wc = nn.Linear(self.encoder_lstm_units, self.decoder_lstm_units)
        self.init_wo = nn.Linear(self.encoder_lstm_units, self.decoder_lstm_units)

        # attention
        self.beta = nn.Parameter(torch.Tensor(self.encoder_lstm_units))
        init.uniform_(self.beta, -1e-2, 1e-2)
        self.W_1 = nn.Linear(self.encoder_lstm_units, self.encoder_lstm_units, bias=False)
        self.W_2 = nn.Linear(self.decoder_lstm_units, self.encoder_lstm_units, bias=False)

        self.W_3 = nn.Linear(self.decoder_lstm_units+self.encoder_lstm_units, self.decoder_lstm_units, bias=False)
        self.W_out = nn.Linear(self.decoder_lstm_units, self.vocab_size, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.uniform = Uniform(0, 1)
    
    def encode(self, imgs):
        encoded_imgs = self.cnn_encoder(imgs)  # [Batchs, encoder_units, H', W']
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1)  # [Batchs, H', W', encoder_units]
        B, H, W, _ = encoded_imgs.shape
        encoded_imgs = encoded_imgs.contiguous().view(B, H*W, -1) # [Batchs, H' x W', encoder_units]
        if True:
            encoded_imgs = self.add_pos_enc(encoded_imgs)
        return encoded_imgs
    
    def add_pos_enc(self, tensor, min_t=1, max_t=1e4):
        """
        Implements the frequency-based positional encoding described
        in `Attention is all you Need
        Parameters
        ----------
        tensor : ``torch.Tensor``
            a Tensor with shape (batch_size, timesteps, hidden_dim).
        min_timescale : ``float``, optional (default = 1.0)
            The largest timescale to use.
        Returns
        -------
        The input tensor augmented with the sinusoidal frequencies.
        """

        _, t_steps, hidden_dim = tensor.size()
        tst_range = torch.arange(0, t_steps, dtype=torch.long, device=tensor.device).data.float() # (t_steps)
        # half for both sin and cos
        n_tsc = hidden_dim//2
        tsc_range = torch.arange(0, n_tsc, dtype=torch.long, device=tensor.device).data.float() # (n_tsc)

        log_tsc_inc = np.log(max_t / min_t) / (n_tsc - 1) 
        inv_tsc = min_t * torch.exp(tsc_range * -log_tsc_inc) # (n_tsc)

        scaled_time = tst_range.unsqueeze(1) * inv_tsc.unsqueeze(0) #(t_steps, n_tsc)
        sinusoids = torch.randn(scaled_time.size(0), 2*scaled_time.size(1), device=tensor.device) # (t_steps, 2 * n_tsc)
        sinusoids[:, ::2] = torch.sin(scaled_time) # even steps
        sinusoids[:, 1::2] = torch.cos(scaled_time) # odd steps
        if hidden_dim % 2 != 0:
            # if the number of dimensions is odd, the cos and sin
            # timescales had size (hidden_dim - 1) / 2, so we need
            # to add a row of zeros to make up the difference.
            sinusoids = torch.cat([sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)

        return tensor + sinusoids.unsqueeze(0)
        
    def init_decoder(self, enc_out):
        """args:
            enc_out: the output of row encoder [B, H*W, C]
          return:
            h_0, c_0:  h_0 and c_0's shape: [B, dec_units]
            init_O : the average of enc_out  [B, dec_units]
            for decoder
        """
        mean_enc_out = enc_out.mean(dim=1)
        h = self._init_h(mean_enc_out)
        c = self._init_c(mean_enc_out)
        init_o = self._init_o(mean_enc_out)
        return (h, c), init_o

    def _init_h(self, mean_enc_out):
        return torch.tanh(self.init_wh(mean_enc_out))

    def _init_c(self, mean_enc_out):
        return torch.tanh(self.init_wc(mean_enc_out))

    def _init_o(self, mean_enc_out):
        return torch.tanh(self.init_wo(mean_enc_out))
    
    def _get_attn(self, encoder_output, h_t):
        """Attention mechanism
        args:
            encoder_output: row encoder's output [B, L=H*W, C]
            h_t: the current time step hidden state [B, dec_units]
        return:
            context: this time step context [B, C]
            attn_scores: Attention scores
        """
        # cal alpha
        alpha = torch.tanh(self.W_1(encoder_output)+self.W_2(h_t).unsqueeze(1))
        alpha = torch.sum(self.beta*alpha, dim=-1)  # [B, L]
        alpha = F.softmax(alpha, dim=-1)  # [B, L]

        # cal context: [B, C]
        # multiply the weights of each batch with the encoded img
        context = torch.bmm(alpha.unsqueeze(1), encoder_output)
        context = context.squeeze(1)
        return context, alpha
    
    def step_decoding(self, hidden_states, output_t, encoder_output, target):
        """Runing one step decoding"""

        prev_y = self.embedding(target).squeeze(1)  # [B, emb_size]
        inp = torch.cat([prev_y, output_t], dim=1)  # [B, emb_size+dec_rnn_h]
        h_t, c_t = self.rnn_decoder(inp, hidden_states)  # h_t:[B, dec_rnn_h]
        h_t = self.dropout(h_t)
        c_t = self.dropout(c_t)

        # context_t : [B, C]
        context_t, attn_scores = self._get_attn(encoder_output, h_t)

        # [B, dec_rnn_h]
        output_t = self.W_3(torch.cat([h_t, context_t], dim=1)).tanh()
        output_t = self.dropout(output_t)

        # calculate logit
        logit = F.softmax(self.W_out(output_t), dim=1)  # [B, out_size]

        return (h_t, c_t), output_t, logit, attn_scores
    
    def forward(self, imgs, formulas, epsilon=1.):
        """args:
        imgs: [B, C, H, W]
        formulas: [B, MAX_LEN]
        epsilon: probability of the current time step to
                 use the true previous token
        return:
        logits: [B, MAX_LEN, VOCAB_SIZE]
        """
        # encoding
        encoded_imgs = self.encode(imgs)  # [B, H*W, 512]
        # init decoder's states
        dec_states, o_t = self.init_decoder(encoded_imgs)
        max_len = formulas.size(1)
        logits = []
        append_output = logits.append
        attention = []
        for t in range(max_len):
            target = formulas[:, t:t+1]
            # schedule sampling
            if logits and self.uniform.sample().item() > epsilon:
                target = torch.argmax(logits[-1], dim=1, keepdim=True)
            # ont step decoding
            dec_states, o_t, logit, attn_scores = self.step_decoding(
                dec_states, o_t, encoded_imgs, target)
            append_output(logit)
            #attention.append(attn_scores)
        logits = torch.stack(logits, dim=1)  # [B, MAX_LEN, out_size]
        return logits #, attention
