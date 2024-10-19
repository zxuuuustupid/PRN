function [  ] = Untitled2( data,path,lable)
%UNTITLED6 此处显示有关此函数的摘要
%   此处显示详细说明

fs=2048; 
%采样频率
num_samples =1000;
resampleing = floor(((length(data))-fs)/(num_samples-1))  

dirname=[path,'\',lable]
mkdir(dirname)
for k=1:num_samples
    k1=(k-1)*resampleing+1;
    k2=(k-1)*resampleing+fs;
    data1=data(k1:k2,:)

    %ylim([20,120])
    %xlabel('时间 t/s')
    %ylabel('频率 f/Hz')fs=320
    %t=0:25.6/320:25.6
    dt=1/fs
    timesrart=0
    timeend=fs/fs
    t=(0:(timeend-timesrart)/dt-1)*dt+timesrart;
    [wt,f,coi] = cwt(double(data1),'morse',fs);
    %f=scal2frq(scales,wavename,1/fs)   
    pcolor(t,f,abs(wt));
      shading flat
   %ylim([10,120])
     %xlim([-0.1,3.3])
    %shading flat
    %title('小波时频图')
    str='train\1.png';
  
    saveas(gcf,str);
    h=imread(str);
    h=imresize(h(113:570,145:760,:),[84,84],'nearest');
    
    imwrite(h,[dirname,'\',lable,'_',num2str(k),'.jpg']);
    
    

   
end
     

end