# Electronic-Music-Assassin
This repository is the core implementation of the black-box audio adversarial attack approach proposed by our paper (Electronic Music Assassin: Towards Imperceptible Physical Adversarial Attacks against Black-box Automatic Speech Recognitions).

# Installation
1.Clone this repo.
```
git clone https://github.com/yzslry/Electronic-Music-Assassin.git
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
You can see the AEs we used in our survey on human auditory perception in the folder `AEs`. AEs for both the physical and digital worlds are included in it.

To facilitate comparison, we split the target commands into two groups, one half for comparison in the physical world and the other half for comparison in the digital world.

We first show five groups of AEs on Google Assistant in the physical world:


https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/701e06d7-e948-4e0f-bac0-0135eddae9bc

https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/b9b2a2b7-5af1-45c2-8f8c-cceb5ed14b0d

https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/173c61cd-7116-4f13-b656-5d26f8d5832d

https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/2b93a0f7-8785-4f5b-a864-a0992646c2f8

https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/0cbe1910-da85-4256-b4d9-d0e12f70276b


Next we present five groups of AEs in the digital world:

https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/7f24c697-2408-448f-a0e4-e09acd213f44

https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/4a6022ff-06f5-49bb-b360-52aa72ff7b19

https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/b03d2a73-b014-4d3f-8fab-5ca56f011713

https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/c136ae05-1b38-4acd-a5ea-adc62b687fb3

https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/da7b119e-1be9-451d-8ffc-97e57159359f

Finally, we show some videos taken during the attack in the physical world and you can see all the videos in the `videos` folder:

https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/2bdf326b-5e73-43a5-88a3-2d1883ea8a93

https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/3b6845a0-acd2-44f2-beb8-6db24d9d07b3

https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/b22908d1-6312-40aa-8ff0-37a3824d95e6

https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/1420fb71-ab14-4fab-9346-7d985a35e184

https://github.com/yzslry/Electronic-Music-Assassin/assets/46239997/b45c3794-9471-44d3-82fe-b3ac6ba72671

