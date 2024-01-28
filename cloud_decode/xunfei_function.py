# -*- coding:utf-8 -*-
#
#   author: iflytek
#
#  本demo测试时运行的环境为：Windows + Python3.7
#  本demo测试成功运行时所安装的第三方库及其版本如下，您可自行逐一或者复制到一个新的txt文件利用pip一次性安装：
#   cffi==1.12.3
#   gevent==1.4.0
#   greenlet==0.4.15
#   pycparser==2.19
#   six==1.12.0
#   websocket==0.2.1
#   websocket-client==0.56.0
#
#  语音听写流式 WebAPI 接口调用示例 接口文档（必看）：https://doc.xfyun.cn/rest_api/语音听写（流式版）.html
#  webapi 听写服务参考帖子（必看）：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=38947&extra=
#  语音听写流式WebAPI 服务，热词使用方式：登陆开放平台https://www.xfyun.cn/后，找到控制台--我的应用---语音听写（流式）---服务管理--个性化热词，
#  设置热词
#  注意：热词只能在识别的时候会增加热词的识别权重，需要注意的是增加相应词条的识别率，但并不是绝对的，具体效果以您测试为准。
#  语音听写流式WebAPI 服务，方言试用方法：登陆开放平台https://www.xfyun.cn/后，找到控制台--我的应用---语音听写（流式）---服务管理--识别语种列表
#  可添加语种或方言，添加后会显示该方言的参数值
#  错误码链接：https://www.xfyun.cn/document/error-code （code返回错误码时必看）
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import websocket
import datetime
import hashlib
import base64
import hmac
import json
import wave
from urllib.parse import urlencode
import time
import ssl
import os
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import argparse

STATUS_FIRST_FRAME = 0  
STATUS_CONTINUE_FRAME = 1  
STATUS_LAST_FRAME = 2  


class Ws_Param(object):
    def __init__(self, APPID, APIKey, APISecret, AudioFile, language):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.AudioFile = AudioFile
        self.language = language

        ###
        # ws.result = '' //
        # ws.success = False //
        # ws.errorMSG = '' //
        ###

        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {"domain": "iat", "language": self.language, "accent": "mandarin", "vinfo": 1, "vad_eos": 10000}

    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        url = url + '?' + urlencode(v)
        return url


def on_message(ws, message):
    try:
        code = json.loads(message)["code"]
        sid = json.loads(message)["sid"]
        if code != 0:
            errMsg = json.loads(message)["message"]
            ws.errorMSG = "sid:%s call error:%s code is:%s" % (sid, errMsg, code) 

        else:
            data = json.loads(message)["data"]["result"]["ws"]
            # print(json.loads(message))
            result = ""
            for i in data:
                for w in i["cw"]:
                    result += w["w"]
            result = result.strip()  
            if len(result) != 0:
                # ws.f.write(result)
                ws.GetMessage = 1
                ws.success = True
                ws.result += (result + ' ')

    except Exception as e:
        print("receive msg,but parse exception:", e)
        ws.errorMSG = "receive msg,but parse exception:" + str(e) 


def on_error(ws, error):
    print("### error:", error)
    ws.errorMSG = "### error:" + str(error)  
    pass


def on_close(ws, *kwargs):
    ws.GetMessage = 0


def on_open(ws):
    def run(*args):
        frameSize = 8000 
        intervel = 0.04 
        status = STATUS_FIRST_FRAME 

        with open(ws.AudioFile, "rb") as fp:  # wsParam.AudioFile
            while True:
                buf = fp.read(frameSize)
                if not buf:
                    status = STATUS_LAST_FRAME
                if status == STATUS_FIRST_FRAME:

                    d = {"common": ws.CommonArgs,
                         "business": ws.BusinessArgs,
                         "data": {"status": 0, "format": ws.sample_rate,
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    d = json.dumps(d)
                    ws.send(d)
                    status = STATUS_CONTINUE_FRAME
                elif status == STATUS_CONTINUE_FRAME:
                    d = {"data": {"status": 1, "format": ws.sample_rate,
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                elif status == STATUS_LAST_FRAME:
                    d = {"data": {"status": 2, "format": ws.sample_rate,
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                    break
                time.sleep(intervel)

        for i in range(ws.max_wait_time):
            time.sleep(0.25)
            if ws.GetMessage == 1:
                break
        ws.close()

    thread.start_new_thread(run, ())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XunFei Parameters Settings')
    parser.add_argument('--APPID', default='f7b05a8e', type=str)
    parser.add_argument('--APIKey', default='7dd26837662c403ad13ee7fd82a39382', type=str)
    parser.add_argument('--APISecret', default='OGZhMzg5MDIyMThiOWM3MzlmNjcwNTU3', type=str)
    parser.add_argument('--wav_folder', default="/home/yxj/Phantom-of-Formants/task/20210515-100song-heygoogle_ibm9-airplanemodeon_V3-gap1p0-voicel3Kto6K-mar50-noover/visqol-airplanemode-5d2661-11100-bw100_300_300_0_0-836856b2acaf52df-10to17/836856b2acaf52df/success/", help='Wav folder\'s absolute path.')
    parser.add_argument('--sample_rate', default=16000, help="SampleRate: 8000/16000")
    parser.add_argument('--language', default='en_us', type=str, help='Chinese:zh_cn / English:en_us')
    args = parser.parse_args()

    assert (args.language == 'en_us' or args.language == 'zh_cn')
    assert (args.sample_rate == 8000 or args.sample_rate == 16000)
    # 测试时候在此处正确填写相关信息即可运行
    time1 = datetime.now()

    wavfiles = [files for files in os.listdir(args.wav_folder) if '.wav' in files]
    if len(wavfiles) == 0:
        print('No wav files in this folder!\n')
        exit()

    f = open(os.path.join(args.wav_folder, "XunFei_Recognition_Result.txt"), "w+")

    for wavfile in wavfiles:
        print('Processing ' + wavfile)
        f.write(wavfile + '\n')

        wsParam = Ws_Param(APPID=args.APPID, APISecret=args.APISecret,
                           APIKey=args.APIKey,
                           AudioFile=os.path.join(args.wav_folder, wavfile),
                           language='en_us')
        websocket.enableTrace(False)
        wsUrl = wsParam.create_url()
        ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
        ws.AudioFile = os.path.join(args.wav_folder, wavfile)
        ws.CommonArgs = wsParam.CommonArgs
        ws.BusinessArgs = wsParam.BusinessArgs
        ws.on_open = on_open

        ws.GetMessage = 0
        ws.max_wait_time = 20 * 4
        ws.result = ''
        ws.errorMSG = None
        ws.success = False
        wavFile = wave.open(os.path.join(args.wav_folder, wavfile))
        sample_rate = wavFile.getframerate()
        ws.sample_rate = "audio/L16;rate=16000" if (sample_rate == 16000) else "audio/L16;rate=8000"
        ws.f = f

        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

        print(ws.success)
        print(ws.result)
        print(ws.errorMSG)

        # break

        print('\n\n')
        f.write('\n\n')

    f.close()

    time2 = datetime.now()
    print(time2 - time1)
