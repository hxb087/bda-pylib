# bda-pylib

python语言书写的bda算子


#安装statsmodels
source /opt/anaconda3/bin/activate python36
pip install statsmodels


#安装talib
ssh 10.60.0.54
mkdir /home/huxb/
cd /home/huxb/
#scp root@10.60.0.53:/home/huxb/ta-lib-0.4.0-src.tar.gz .

source /opt/anaconda3/bin/activate python36
sudo wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz  #下载文件
sudo tar -xzf ta-lib-0.4.0-src.tar.gz  #解压

cd ta-lib
./configure --prefix=/usr
make
make install

echo "export LD_LIBRARY_PATH=/lib" >> /etc/profile
source /etc/profile

echo "/usr/lib/" >> /etc/ld.so.conf
ldconfig

pip install ta-lib -U 

#安装pycasso
pip install pycasso
