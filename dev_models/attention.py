
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.utils.model_zoo as model_zoo
import pdb

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
        )

    def forward(self, x):
        #[16, 3, 224, 224]
        x = self.proj(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)[16, 768, 14, 14]
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)[16, 768, 196]
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)[16, 196, 768]

        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (q @ k_t) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x


class A_Block(nn.Module):

    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim,n_heads=n_heads,qkv_bias=qkv_bias,attn_p=attn_p,proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)

    def forward(self, x):
        # print(self.attn(self.norm1(x)))
        x = x + self.attn(self.norm1(x))

        return x

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation
 
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
 
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
 
        out = self.gamma*out + x
        return out,attention


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class Dilations(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(Dilations, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=3,dilation=2,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion, 1024)

        self.patch_embed = PatchEmbed(img_size=224,patch_size=16,in_chans=3,embed_dim=768)
        self.A_blocks = nn.ModuleList([A_Block(dim=768,n_heads=12,mlp_ratio=4.,qkv_bias=True,p=0.,attn_p=0.)])
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.head = nn.Linear(768, 1024)
        # self.Atten = Self_Attn(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 第一种self-attention，已经注释
        # # print(x.shape,'1')
        # x1 = self.patch_embed(x)#[16, 196, 768]
        # # print(x1.shape,'2')
        # for A_block in self.A_blocks:
        #     x1 = A_block(x1)#[16, 196, 768]
        #     # print(x1.shape,'3')
        # x1 = self.norm(x1)#[16, 196, 768]
        # # print(x1.shape,'4')
        # x1 = F.softmax(x1, dim = 2)#[16, 196, 768]
        # # print(x1.shape,'5')
        # cls_token_final = x1[:, 0]
        # x1 = self.head(cls_token_final)#[16, 1024]
        # # print(x1.shape,'6')
        # # x1 = relu(x1)
        
        x = self.conv1(x)#[16, 3, 224, 224] [16, 64, 112, 112]
        x = self.bn1(x)#[16, 3, 224, 224] [16, 64, 112, 112]
        x = self.relu(x)#[16, 3, 224, 224] [16, 64, 112, 112]
        x = self.maxpool(x)#[16, 64, 112, 112] [16, 64, 56, 56]
        # pdb.set_trace()
        x = self.layer1(x)#[16, 256, 56, 56]
        x = self.layer2(x)#[16, 512, 28, 28]
        # 第二种self-attention，两种用法类似，可以替换
        out,attention_map = self.Atten(x)# [16,784,784]
        cls_token_final = attention_map[:, 0]
        x1 = self.head(cls_token_final)#[16, 1024]

        # x = self.layer3(out)#[16, 1024, 14, 14]
        # x = self.layer4(x)#[16, 2048, 7, 7]

        # x = self.avgpool(x)#[16, 2048, 1, 1]
        # x = x.view(x.size(0), -1)#[16, 2048]
        # x = self.relu(x)#[16, 2048]

        # x = self.fc1(x)#[16, 1024]
        # x = self.relu(x)#[16, 1024]
        # pdb.set_trace()
        return x1


def SeAtten(**kwargs):
    model = Dilations(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
