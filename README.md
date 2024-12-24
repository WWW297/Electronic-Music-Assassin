# Electronic-Music-Assassin
This repository is the core implementation of the black-box audio adversarial attack approach proposed by our paper (Electronic Music Assassin: Towards Natural Physical Adversarial Attacks against Black-box Automatic Speech Recognitions).

# Installation
1.Clone this repo.
```
git clone https://github.com/Cybersecurity-Electronic-Music-Assassin/Electronic-Music-Assassin.git
cd Electronic-Music-Assassin
```
2.Create a virtual environment running Python 3.8 interpreter of later.

3.Install the dependencies.
```
pip install -r requirements.txt
```
# Usage
1.Register the target ASR cloud services provided and fill in the relevant information in the  `account.py`.

2.Create the `music` folder and add the song you would like as carrier audios to it in wav format with the sample rate of 16000

3.Use the cloud text-to-speech service to generate audios of the target attack commands in wav format with the sample rate of 16000 and place them under the `command` folder

4.For attacks in the digital world, you can obtain the AEs by executing the following code：
```
python attack_digital.py --speech-file-path target_TTS_path --music-file-path ./music --attack-target [tencentyun,aliyun,iflytec,google,azure] --sample-num successful_AEs_number
```
Successful AEs will be saved in folder `success_digital_samples`

5.For attacks in the physical world, you can obtain the AEs by executing the following code：
```
python attack_physical.py --speech-file-path target_TTS_path --music-file-path ./music --sample-num AEs_number
```
AEs will be saved in folder `physical_samples` and you need the test wheather they are successful in the physical devices.

# Demos
You can see the AEs we used in our user study on human auditory perception in the folder `AEs`. AEs for both the physical and digital worlds are included in it. To minimize the impact of player quality and ambient noise on the actual perception, we recommend listening to these demos using headphones.

To facilitate comparison, we split the target commands into two groups, one half for comparison in the physical world and the other half for comparison in the digital world.

We first show five groups of AEs on Google Assistant in the digital world:

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/bef9f92d-0bb0-488b-bbd6-6d96b1c59f84

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/897e5909-a4f6-4ce7-8fb5-ba73c914f5e0

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/861736c1-95e8-44d5-8ca1-1d7461bb85a6

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/125df06a-a7c0-4e97-8813-016891ef040d

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/a1e9ba33-162e-4b06-b252-122170cbfcf1



Next we present five groups of AEs in the physical world:

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/71214dff-5e78-4f6d-9021-571a8bb8eaec

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/08b8da7d-a22a-4f5c-b088-2d16c9cc2468

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/8962286b-5075-4aec-9735-697f600d8b7e

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/92bed9f4-0c22-4d6f-9856-b036c729658a

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/2ed19fda-e55c-47e8-9ca0-7e71570bcf0f


Then, we show some videos taken during the attack in the physical world:


https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/e9e6b4ba-c7bb-421f-9e9d-6f1c61c1bafe

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/91e00a30-accd-431b-8837-a6843487047e

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/856453aa-67aa-422e-bc5c-56ecbee45dc1

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/c1b52470-35ed-40d3-a9cb-d7a555b9e1a2

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/e05c09e7-ba2c-4b69-8ba4-9458a8d9a865

Finally, we present a demo that automatically wakes up Google Assistant and successfully executes an attack：

https://github.com/NDSS279/Electronic-Music-Assassin/assets/157971566/9d36565c-2ee2-4504-9fe0-80adf7c7e62d

