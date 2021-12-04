import torch 
import torch.nn as nn 

class SelfAttention(nn.Module):
    def __init__(
        self, 
        embedding_size, 
        heads
    ) -> None:

        super(SelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.heads = heads 
        self.head_dims = embedding_size // heads 

        self.values = nn.Linear(self.head_dims, self.head_dims, bias=False)
        self.keys = nn.Linear(self.head_dims, self.head_dims, bias=False)
        self.queries = nn.Linear(self.head_dimsm, self.head_dims, bias=False)

        self.fc_out = nn.Linear(self.heads * self.head_dims, self.embedding_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embeddings into head pieces 
        values = values.reshape(N, value_len, self.heads, self.head_dims)
        keys = keys.reshape(N, key_len, self.heads, self.head_dims)
        queries = query.reshape(N, key_len, self.heads, self.head_dims)

        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])
      
        # QUERY SHAPE = (N, query_len, heads, head_dims)
        # KEYS SHAPE = (N, key_len, heads, head_dims)
        # ENERGY SHAPE = (N, heads, query_len, key_len)

        if mask is not None:
            # Mask of target is triangular matrix
            energy = energy.masked_fill(mask == 0, float("-1e20")) # If element of mask is 0 then we shut that off
        
        # Run through softmax 
        attention = torch.softmax(energy / (self.embedding_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql", "nlhd->nqhd", [attention, values]).reshape(N, query_len, self.head*self.head_dims)
        # ATTENTION SHAPE = (N, heads, query_len, key_len)
        # VALUES SHAPE = (N, value_len, heads, head_dims)
        # After einsum: (N, query_len, heads, head_dims)          ( After multiplication) -> Flattened last 2 dims

        out = self.fc_out(out)
        return out 
    

class TransformerBlock(nn.Module):

    def __init__(
        self, 
        embedding_size, 
        heads, 
        dropout, 
        forward_exp
    ) -> None:
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_size, heads)
        self.norm_1 = nn.LayerNorm(embedding_size)
        self.norm_2 = nn.LayerNorm(embedding_size)

        self.net = nn.Sequential(
            nn.Linear(embedding_size, forward_exp * embedding_size),
            nn.ReLU(), 
            nn.Linear(forward_exp * embedding_size, embedding_size)
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        X = self.dropout(self.norm_1(attention + query))
        forward = self.net(X)
        out = self.dropout(self.norm_2(forward + X))
        return out 
    
    
class DecoderBlock(nn.Module):
    def __init__(
        self, 
        embedding_size, 
        heads, 
        forward_exp, 
        dropout, 
    ) -> None:

        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embedding_size, heads)
        self.norm = nn.LayerNorm(embedding_size)
        
        self.transformer_block = TransformerBlock(
            embedding_size, heads, dropout, forward_exp
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)

        return out 
    

