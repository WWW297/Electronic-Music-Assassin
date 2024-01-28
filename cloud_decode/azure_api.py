# coding=utf-8
import glob

import azure.cognitiveservices.speech as speech_sdk
import click

from utils import *
from account import ACCOUNT

__api_key__ = ACCOUNT["Microsoft"]["api_key"]
__api_region__ = ACCOUNT["Microsoft"]["location"]


def azure_decode_multi(audio_files: List[str], key: str = __api_key__, region: str = __api_region__, save_result: bool = True, max_workers: int = None, max_repeat_time: int = 1, wait_seconds: int = 10, output_format="transaction", language="zh-CN") -> List[str]:
    """
    using multi-process to decode audio files asynchronously.
    Because of the CancelException of cortana api, you can use max_repeat_time to control the re-decode times, and use wait_seconds in seconds to control the wait time between two queries.
    :return: transactions of every audio file.
    """
    with futures.ProcessPoolExecutor(max_workers=max_workers) as _executor_:
        jobs = []
        for audio_file in audio_files:
            jobs.append(
                _executor_.submit(
                    azure_decode, audio_file, key, region, save_result, max_repeat_time, wait_seconds, output_format, language
                )
            )
        results = wait_for_jobs(jobs, _executor_, "[Azure] Decode Sample Progress: ")
    return results


def azure_find(json_file: str, expected_string: str, delete_stop_words: bool = False) -> Tuple[bool, Union[str, None], Union[float, None]]:
    with open(json_file, 'r') as _file_:
        decode_result = json.load(_file_)

    did_find, decode_string, find_confidence = False, None, None
    if decode_result['success'] is True and isinstance(decode_result['result'], dict):
        result = decode_result['result']
        n_best = result['NBest']
        for result in n_best:
            display = check_transaction(result['Display'], delete_stop_words)
            split_display = display.split()
            confidence = result['Confidence']
            if all([expected_word in split_display for expected_word in expected_string.split()]):
                did_find = True
                if find_confidence is None or find_confidence < confidence:
                    find_confidence = confidence
                    decode_string = display

    return did_find, decode_string, find_confidence


def azure_result_2_prob(json_result, transaction, _delete_stop_words_: bool = False, _error_prob_: float = float('-inf')) -> float:
    transaction = check_transaction(transaction, _delete_stop_words_)

    if not json_result['success']:
        return _error_prob_
    result = json_result['result']
    if result == "Miss Error" or not isinstance(result, dict):
        return _error_prob_
    n_best = result['NBest']

    _is_transaction_in_ = False
    max_prob = 0.0
    for k_result in n_best:
        confidence = k_result['Confidence']
        display = k_result['Display']
        display = check_transaction(display, _delete_stop_words_)
        if transaction in display:
            _is_transaction_in_ = True
            max_prob = max(confidence, max_prob)

    if _is_transaction_in_:
        return max_prob
    else:
        return _error_prob_


# todo: 开源的时候隐藏key
@exception_printer
def azure_decode(audio_file: str, key: str = __api_key__, region: str = __api_region__, save_result: bool = False, _max_repeat_time_: int = 10, _wait_seconds_: int = 10, language="en-US"):
    speech_config = speech_sdk.SpeechConfig(subscription=key, region=region)
    speech_config.output_format = speech_sdk.OutputFormat.Detailed

    try:
        wav_result = {"success": False}
        _flag_ = True
        repeat_time = 0
        audio_config = speech_sdk.AudioConfig(filename=audio_file)
        speech_recognizer = speech_sdk.SpeechRecognizer(speech_config, audio_config, language=language)
        while _flag_ and repeat_time < _max_repeat_time_:
            _flag_ = False
            repeat_time += 1
            response = speech_recognizer.recognize_once()
            if response.reason == speech_sdk.ResultReason.RecognizedSpeech:
                wav_result = {
                    "success": True,
                    "result": json.loads(response.json)['DisplayText']
                }
            elif response.reason == speech_sdk.ResultReason.NoMatch:
                wav_result = {
                    "success": True,
                    "result": "No Match"
                }
            elif response.reason == speech_sdk.ResultReason.Canceled:
                _flag_ = True
                if repeat_time < _max_repeat_time_:
                    time.sleep(_wait_seconds_)  # wait for 30 seconds
                    logger.warning("Decode wave {} failed. Repeat Again {}.".format(audio_file, repeat_time))
                else:
                    logger.warning("Wav {} recognition canceled: {}".format(audio_file, response.cancellation_details.reason))
                    wav_result = {
                        "success": False,
                        "result": "Canceled"
                    }
            else:
                logger.error("Azure recognize wav file {} failed: {}".format(audio_file, response.reason))
                wav_result = {
                    "success": False,
                    "result": response.reason
                }
                print(response.reason)
    except InterruptedError as _err_:
        raise _err_
    except Exception:
        traceback.print_exc()
        logger.error("Azure recognize wav file {} failed.".format(audio_file))
        wav_result = {
            "success": False,
            "result": "Miss Error"
        }

    if save_result:
        audio_folder = os.path.dirname(audio_file)
        wav_name = os.path.splitext(os.path.basename(audio_file))[0]
        wav_decode_file = os.path.join(audio_folder, "{}{}".format(wav_name, AZURE_JSON_SUFFIX))
        with open(wav_decode_file, 'w') as _file_:
            json.dump(wav_result, _file_, ensure_ascii=False)

    return wav_result['result'],wav_result['success']


