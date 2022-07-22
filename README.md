1. install driver
cd driver
cd seeed-vociecard
sudo ./instal.sh
sudo reboot

check wheter exist driver
arecord -l 

2. set ip
vi main.py
revies ip and port

3. start siren detection
chmod 777 ./start.sh
./start.sh
