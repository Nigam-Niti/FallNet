class SupervisionNet(nn.Module):
  def __init__(self):
      super(SupervisionNet, self).__init__()
      self.base_model = nn.Sequential(*list(i3d_Mixed_5c.children())[0:])
      self.avg_pool =  nn.AvgPool3d(kernel_size=[1, 7, 7], stride=(1, 1, 1))
      self.relu = nn.ReLU(inplace=True) 
      self.point_conv = nn.Conv1d(in_channels= 1024, out_channels=256, kernel_size=[1], stride=(1))
      self.dropout = nn.Dropout2d(0.3)
      self.fine_conv = nn.Conv1d(256, 2 * 51, kernel_size = 1, stride = 1)
      self.cross_conv = nn.Conv1d(2 * 51,  51, kernel_size = 1, stride = 1)
      self.cross_pool = nn.AvgPool1d(2, stride = 2, padding = 0)                                       
      #self.fc1 = nn.Linear(256, 51)

  def forward(self, x):
      out = self.base_model(x)
      #print("size after pretrained model==> ", out.size()) #torch.Size([16, 1024, 2, 7, 7])
      out = self.avg_pool(out) #torch.Size([16, 1024, 2, 1, 1])
      out = self.relu(out).squeeze(-1).squeeze(3) #torch.Size([16, 1024, 2])
      out = self.point_conv(out) #torch.Size([16, 256, 2])
      out = self.dropout(out)
      out = self.fine_conv(out) #torch.Size([16, 102, 2])
      out = self.cross_conv(out) #torch.Size([16, 51, 2])
      out = self.cross_pool(out).squeeze(-1) #torch.Size([16, 51, 1])   
      #print("size after cross_pool model==> ", out.size()) #torch.Size([16, 51])
      out = torch.log_softmax(out, dim=1)#torch.Size([16, 51])
      #print(out)
      return out
