##空洞卷积神经网络的搭建
class MyConvdilaNet(nn.Module):
    def __init__(self):
        super(MyConvdilaNet,self).__init__()
        ##定义第一个卷积层
        self.conv1 = nn.Sequential(
            ##卷积后： （1*28*28）---（16*28*28）
            nn.Conv2d(1,16,3,1,1,dilation = 2),
            nn.ReLU(),
            nn.AvgPool2d(2,2), ##(16*26*26)--(16*13*13)
        )
        ##定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,0,dilation=2),
            ##卷积操作 （16*13*13）---（32*9*9）
            nn.ReLU(),
            nn.AvgPool2d(2,2),
        )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(32*4*4,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    ##定义网络的向前传播路径
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1) ##展平多维的卷积图层
        return x

