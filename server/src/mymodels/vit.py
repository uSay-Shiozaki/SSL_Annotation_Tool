import math
import torch
import torch.nn as nn
import sys
sys.path.append('/mnt/home/irielab/workspace/projects/my_research')

from functools import partial
from ibot_main.utils import trunc_normal_

# We probably can use nn.Dropout()
def dropOutPath(x, dropOutProb: float = 0., training: bool = False):
  if dropOutProb == 0. or not training:
    return x
  
  keepProb: float = 1. - dropOutProb
  # (a,) means a tuple with a value, not (a)
  shape: tuple = (x.shape[0],) + (1,) * (x.ndim - 1) # work with diff dim tensors, not just 2D ConvNets
  # mask
  randomTensor = keepProb + torch.rand(shape, dtype=x.dtype, device=x.device)
  randomTensor.floor_() #prob is 0 < x < 1, so return 0 or 1 # floor_() is in-place version of floor()
  # divided by keepProb and multiply with randomTensor
  # Low keepProb makes high prob in tensor.
  # Fewer nodes make high prob.
  output = x.div(keepProb) * randomTensor
  return output

class DropPath(nn.Module):
  
  def __init__(self, dropOutProb=None):
    super().__init__()
    self.dropOutProb = dropOutProb
    
  def forward(self, x):
    #XXX Where self.trainig?
    return dropOutPath(x, self.dropOutProb, self.training)
  
class Mlp(nn.Module):
  def __init__(self, inFeatures: int,
              hiddenFeatures: int =None,
              outFeatures: int = None,
              actLayer: nn.Module= nn.GELU,
              dropOutProb: float=0.):
    
    super().__init__()
    hiddenFeatures = hiddenFeatures or inFeatures
    outFeatures = outFeatures or inFeatures
    # fc: Fully Connected Layer
    self.fc1 = nn.Linear(inFeatures, hiddenFeatures)
    self.act = actLayer()
    self.fc2 = nn.Linear(hiddenFeatures, outFeatures)
    self.dropOut = nn.Dropout(dropOutProb)
    
  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.dropOut(x)
    x = self.fc2(x)
    x = self.dropOut(x)
    return x
  
class Attention(nn.Module):
  def __init__(self, dim: int,
              numHeads: int = 8,
              qkvBias: bool =False,
              qkScale: float = None,
              attnDropProb: float = 0.,
              projDropProb: float = 0.):
    
    super().__init__()
    self.numHeads = numHeads
    #TODO Check dim (dim // num_heads) ** -0.5
    self.scale = qkScale or (dim // numHeads) ** -0.5
    self.qkv = nn.Linear(dim, dim * 3, bias=qkvBias)
    self.attnDrop = nn.Dropout(attnDropProb)
    self.proj = nn.Linear(dim, dim)
    self.projDrop = nn.Dropout(projDropProb)
    self.softMax = nn.Softmax(dim = -1)
    
  def forward(self, x):
    # Batch, The No. of sumples, Channel
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.numHeads, C // self.numHeads).permute(2, 0, 3, 1, 4) 
    # C // self.numHeads: the No. of patches through Channels
    # (3, B, self.numHeads, N, C // self.numHeads)
    q, k, v = qkv[0], qkv[1], qkv[2]
    # @: matrix multiplication
    attn = (q @ k.transpose(-2, -1)) * self.scale
    # original: attn.softmax(dim = -1)
    attn = self.softMax(attn)
    attn = self.attnDrop(attn)
    
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, attn
  
class Block(nn.Module):
  def __init__(self, dim: int, 
              numHeads: int,
              mlpRatio: float = 4.,
              qkvBias: bool = False,
              qkScale: float = None,
              projDropProb: float = 0.,
              attnDropProb: float = 0.,
              dropPathProb: float = 0.,
              actLayer: nn.Module = nn.GELU,
              normLayer: nn.Module = nn.LayerNorm,
              initValues: int = 0):
    
    super().__init__()
    self.norm1 = normLayer(dim)
    self.attn = Attention(
      dim, numHeads=numHeads, qkvBias=qkvBias,
      qkScale=qkScale, attnDropProb=attnDropProb, projDropProb=projDropProb
    )
    self.dropPath = DropPath(dropPathProb) if dropPathProb > 0. else nn.Identity()
    self.norm2 = normLayer(dim)
    self.mlp = Mlp(
      inFeatures=dim, hiddenFeature=int(dim * mlpRatio),
      actLayer = actLayer, dropOutProb = dropPathProb
    )
    
    if not initValues > 0:
      self.gamma1, self.gamma2 = None, None
    else:
      # torch.ones: make tensor fullfilled values 1
      self.gamma1 = nn.Parameter(initValues * torch.ones((dim)), requires_grad=True)
      self.gamma2 = nn.Parameter(initValues * torch.ones((dim)), requires_grad=True)
  
  def forward(self, x, returnAttention=False):
    y = self.norm1(x)
    y, attn = self.attn(y)
    if returnAttention:
      return attn
    if self.gamma1 is None:
      # (x + ): skip connection
      x = x + self.dropPath(y)
      x = self.norm2(x)
      x = self.mlp(x)
      x = x + self.dropPath(x)
    else:
      x = self.gamma1 * y
      x = x + self.dropPath(x)
      x = self.norm2(x)
      x = self.gamma2 * self.mlp(x)
      x = x + self.dropPath(x)
    return x

class PatchEmbed(nn.Module):
  def __init__(self, imgSize:int = 224,patchSize:int = 16,
              inChans: int = 3, embedDim: int = 768):
    super().__init__()
    numPatches = (imgSize // patchSize) * (imgSize // patchSize)
    self.imgSize = imgSize
    self.patchSize = patchSize
    self.numPatches = numPatches
    
    self.proj = nn.Conv2d(inChans, embedDim,
                          kernelSize=patchSize, stride=patchSize)
  
  def forward(self, x):
    B, C, H, W = x.shape
    return self.proj(x)
  
class VisionTransformer(nn.Module):
  def __init__(self, imgSize: int = 224,
              patchSize: int = 16, inChannels: int = 3,
              numClasses: int = 0, embedDim: int = 768,
              depth: int = 12, numHeads: int = 12,
              mlpRatio: float = 4., qkvBias: bool = False,
              qkScale: float = None, dropRate: float = 0.,
              attnDropProb: float = 0., dropPathProb: float = 0.,
              normLayer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
              returnAllTokens: bool = False, initValues: int = 0,
              useMeanPooling: bool = False, maskedImgModeling: bool = False):
    super().__init__()
    self.numFeatures = self.embedDim = embedDim
    self.returnAllTokens = returnAllTokens
    self.patchEmbed = PatchEmbed(
      imgSize=imgSize, patchSize=patchSize, inChannels=inChannels, embedDim=embedDim
    )
    numPatches = self.patchEmbed.numPatches
    
    self.clsToken = nn.Parameter(torch.zeros(1, 1, embedDim))
    self.posEmbed = nn.Parameter(torch.zeros(1, numPatches + 1, embedDim))
    self.posDrop = nn.Dropout(p=dropPathProb)
    depthDecayRule = [x.item() for x in torch.linspace(0, dropPathProb, depth)]
    self.blocks = nn.ModuleList([
      Block(
        dim=embedDim, numHeads=numHeads, mlpRatio=mlpRatio,
        qkvBias=qkvBias, qkScale=qkScale,
        dropPathProb=dropRate, attnDropProb=attnDropProb, 
        dropPath=depthDecayRule[i], normLayer=normLayer,
        initValues=initValues
      ) for i in range(depth)
    ])
    
    self.norm == nn.Identity() if useMeanPooling else normLayer(embedDim)
    self.fcNorm = normLayer(embedDim) if useMeanPooling else None
    # Classifier head
    self.head = nn.Linear(embedDim, numClasses) if numClasses > 0 else nn.Identity()
    
    # truncate Nomalization in util.py
    truncNormal(self.posEmbed, std=.02)
    truncNormal(self.clsToken, std=.02)
    self.apply(self._initWeights)
    
    # masked image modeling
    self.maskedImgModeling = maskedImgModeling
    if maskedImgModeling:
      self.maskedEmbed = nn.Parameter(torch.zeros(1, embedDim))
    
  def _initWeights(self, m):
    if isinstance(m, nn.Linear):
      truncNormal(m.weight, std=.02)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)
      
  def interpolatePosEncoding(self, x, w, h):
    nPatch = x.shape[1] - 1
    N = self.posEmbed.shape[1] - 1
    if nPatch == N and w == h:
      return self.posEmbed
    classPosEmbed = self.posEmbed[:, 0]
    patchPosEmbed = self.posEMbed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patchEmbed.patchSize
    h0 = h // self.patchEmbed.patchSize
    
    # iBOT author add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    
    w0, h0 = w0 + 0.1, h0 + 0.1
    # resize tensor
    
    #TODO check following calculation
    patchPosEmbed = nn.functional.interpolate(
      patchPosEmbed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
      scaleFactor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
      mode='bicubic'
    )
    
    #TODO check meaning of this condition
    assert int(w0) == patchPosEmbed.shape[-2] and int(h0) == patchPosEmbed.shape[-1]
    
    #TODO check following conversion
    patchPosEmbed = patchPosEmbed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((classPosEmbed.unsqueeze(0), patchPosEmbed), dim=1)
  
  def prepareTokens(self, x, mask=None) -> nn.Module:
    B, NoC, W, H = x.shape
    
    # patch linear embedding
    x = self.patchEmbed(x)
    
    # mask image modeling
    if mask is not None:
      x = self.maskModel(x, mask)
    x = x.flatten(2).transpose(1, 2)
    
    # add cls token to embed patch tokens
    clsTokens = self.clsToken.expand(B, -1, -1)
    x = torch.cat((clsTokens, x), dim=1)
    
    # add positional encoding to each token
    x = x + self.interpolatePosEncoding(x, W, H)
    
    return self.posDrop(x)
  
  def forward(self, x, returnAllTokens=None, mask=None):
    # Masked Image Modeling: MIM
    if self.maskedImgModeling:
      assert mask is not None
      x = self.prepareTokens(x, mask=mask)
    else:
      x = self.prepareTokens(x)
      
    for block in self.blocks: # nn.ModuleList[depth]
      x = block(x)
    
    x = self.norm(x) # nn.Identity or self.normLayer()
    if self.fcNorm() is not None: # triggerd useMeanPooling
      x[:, 0] = self.fcNorm(x[:, 1:, :].mean(1))
    
    returnAllTokens = self.returnAllTokens if returnAllTokens is None \
    else returnAllTokens
    
    return x if returnAllTokens else x[:, 0]
  
  def getLastSelfAttention(self, x):
    x = self.prepareTokens(x)
    for i, block in enumerate(self.blocks):
      if i < len(self.blocks) - 1:
        x = block(x)
      else:
        # return attention of the last block
        return block(x, returnAttention=True)
  
  def getIntermediateLayers(self, x, n=1):
    x = self.prepareTokens(x)
    
    # return the output tokens from the 'n' last blocks
    output = []
    for i, block in enumerate(self.blocks):
      x = block(x)
      if len(self.blocks) - i <= n:
        output.append(self.norm(x)) #XXX need norm layer?
    return output

  def getNumLayers(self):
    return len(self.blocks)
  
  def maskModel(self, x, mask):
    #TODO check the actual value of maskedEmbed // 0 or 1 ?
    x.permute(0, 2, 3, 1)[mask, :] = self.maskedEmbed.to(x.dtype) # nn.Parameter()
    return x
    
def vitSmall(patchSize=16, **kwargs):
  model = VisionTransformer(
    patchSize=patchSize, embedDim=384, depth=12, numHeads=6, mlpRatio=4,
    qkvBias=True, **kwargs
  )
  return model

def vitBase(patchSize=16, **kwargs):
  model = VisionTransformer(
    patchSize=patchSize, embedDim=768, depth=12, numHeads=12, mlpRatio=4,
    qkvBias=True, **kwargs
  )
  return model