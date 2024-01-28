# coding=utf-8
import glob
import logging
import sys
import traceback
from cloud_decode.xunfei_function import *
from typing import List, Union, Tuple
from cloud_decode.log_utils import exception_printer, get_logger
from account import ACCOUNT

__app_id__ = ACCOUNT["iFLYTEK"]["app_id"]
__api_key__ = ACCOUNT["iFLYTEK"]["app_key"]
__api_secret__ = ACCOUNT["iFLYTEK"]["app_secret"]

IFLYTEC_JSON_SUFFIX = ".iflytec.json"


def xunfei_decode_multi(audio_files: List[str], appid: str = __app_id__, apikey: str = __api_key__, apisecret: str = __api_secret__, save_result: bool = True, output_format="transaction", language="en_us", re_decode_failed: bool = False) -> List[str]:
    """
    using multi-process to decode audio files asynchronously.
    Because of the CancelException of cortana api, you can use max_repeat_time to control the re-decode times, and use wait_seconds in seconds to control the wait time between two queries.
    :return: transactions of every audio file.
    """
    if re_decode_failed == True:
        Temp_audio_files = []
        for audio_file in audio_files:
            if os.path.exists(audio_file.replace('.wav', IFLYTEC_JSON_SUFFIX)) == False:
                Temp_audio_files.append(audio_file)

        if len(Temp_audio_files) == 0:
            print('All wav files are already decoded!\n')
            sys.exit()

        audio_files = Temp_audio_files

    results = []
    for audio_file in audio_files:
        print(audio_file.split('/')[-1])
        result = xunfei_decode(audio_file, appid, apikey, apisecret, language=language)
        results.append(result)
    return results


@exception_printer
def xunfei_decode(audio_file: str, appid: str = __app_id__, apikey: str = __api_key__, apisecret: str = __api_secret__, save_result: bool = False, _wait_seconds_: int = 60, language="en_us", logger: logging.Logger = get_logger()):
    try:
        # print('Processing '+audio_file)
        wsParam = Ws_Param(APPID=appid, APISecret=apisecret, APIKey=apikey, AudioFile=audio_file, language=language)
        websocket.enableTrace(False)
        wsUrl = wsParam.create_url()
        ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
        ws.AudioFile = wsParam.AudioFile
        ws.CommonArgs = wsParam.CommonArgs
        ws.BusinessArgs = wsParam.BusinessArgs

        ws.on_open = on_open

        ws.GetMessage = 0
        ws.result = ''
        ws.errorMSG = None
        ws.success = False
        ws.max_wait_time = _wait_seconds_ * 4
        wavFile = wave.open(audio_file)
        sample_rate = wavFile.getframerate()
        ws.sample_rate = "audio/L16;rate=16000" if (sample_rate == 16000) else "audio/L16;rate=8000"

        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})  # , ping_timeout=ws.max_wait_time

        wav_result = {
            "success": True,
            "errorMSG": ws.errorMSG,
            "result": ws.result,
            "confidence": None
        }
    except InterruptedError as _err_:
        raise _err_
    except Exception:
        traceback.print_exc()
        logger.error("XunFei recognize wav file {} failed.".format(audio_file))

        wav_result = {
            "success": False,
            "errorMSG": "Miss Error",
            "result": "Miss Error",
            "confidence": None
        }

    if save_result:
        with open(audio_file.replace('.wav', IFLYTEC_JSON_SUFFIX), 'w') as _file_:
            json.dump(wav_result, _file_, ensure_ascii=False)

    return wav_result['result'],wav_result['success']

def api_recognize(logger=get_logger()):
    logger.info("Start decode folder {} using XunFei.\n".format(args.wav_folder))
    wav_files = glob.glob(os.path.join(args.wav_folder, "**", "*.wav"), recursive=True)
    xunfei_decode_multi(wav_files, appid=__app_id__, apikey=__api_key__, apisecret=__api_secret__, re_decode_failed=args.re_decode_failed,language=args.language)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Google Speech to Text.')
    parser.add_argument('wav_folder', type=str, help="Where is the wav_folder.")
    parser.add_argument('--re_decode_all', action="store_true", default=False, help="Whether re-decode all the wave files. Default is False.")
    parser.add_argument('--re_decode_failed', action="store_true", default=False, help="Whether to re-decode the failed decoding wave files. Default is False.")
    parser.add_argument("--language", "-l", default='en_us', choices=['en-us', 'zh-cn'])
    args = parser.parse_args()

    api_recognize()

