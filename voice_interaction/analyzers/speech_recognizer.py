"""
语音识别模块

提供基于百度语音API的语音识别功能，包含令牌管理、音频校验和错误重试。
"""

import json
import base64
import urllib.request
import urllib.error
import time
from typing import Optional, Dict, Any
import speech_recognition as sr
from ..config import SPEECH_CONFIG, BAIDU_API_KEY, BAIDU_SECRET_KEY


class SpeechRecognizer:
    """语音识别器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化语音识别器

        参数:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or SPEECH_CONFIG.copy()
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = self.config['pause_threshold']
        self.recognizer.non_speaking_duration = self.config['non_speaking_duration']
        self._token: Optional[str] = None
        self._last_token_time: float = 0
        self._token_validity: int = 2592000  # 百度 token 有效期 30 天（秒）

    def get_baidu_token(self, force_refresh: bool = False) -> str:
        """
        获取百度语音API的访问令牌（带缓存和刷新机制）

        参数:
            force_refresh: 是否强制刷新令牌

        返回:
            访问令牌字符串

        异常:
            RuntimeError: 当 API 密钥缺失或请求失败时
        """
        if not BAIDU_API_KEY or not BAIDU_SECRET_KEY:
            raise RuntimeError("百度 API 密钥未配置，请设置 BAIDU_API_KEY 和 BAIDU_SECRET_KEY")

        # 检查是否需要刷新
        current_time = time.time()
        if (not force_refresh and
                self._token and
                (current_time - self._last_token_time) < self._token_validity):
            return self._token

        try:
            url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={BAIDU_API_KEY}&client_secret={BAIDU_SECRET_KEY}"
            req = urllib.request.Request(url, method='POST')
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())

            if 'access_token' not in result:
                raise RuntimeError(f"百度 token 获取失败: {result.get('error_description', '未知错误')}")

            self._token = result['access_token']
            self._last_token_time = current_time
            return self._token

        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            raise RuntimeError(f"获取百度 token 失败: {e}")

    def recognize_speech_baidu(self, audio_data: sr.AudioData) -> str:
        """
        使用百度语音API识别语音

        参数:
            audio_data: 音频数据（来自 speech_recognition）

        返回:
            识别的文本字符串（可能为空）

        异常:
            ValueError: 音频过长
            RuntimeError: API 调用失败
        """
        # 获取有效 token
        if not self._token:
            self.get_baidu_token()

        rate = self.config['sample_rate']
        raw_data = audio_data.get_wav_data(convert_rate=rate, convert_width=2)

        # 音频长度校验
        duration_sec = len(raw_data) / (rate * 2)
        if duration_sec > self.config['max_audio_length']:
            raise ValueError(f"音频过长（{duration_sec:.1f}秒 > {self.config['max_audio_length']}秒），请精简回答")

        # 构建请求
        body_dict = {
            "format": "pcm",
            "rate": rate,
            "dev_pid": 1537,  # 中文普通话
            "channel": 1,
            "token": self._token,
            "cuid": "voice_interaction",
            "len": len(raw_data),
            "speech": base64.b64encode(raw_data).decode()
        }
        body = json.dumps(body_dict).encode()

        try:
            req = urllib.request.Request(
                "http://vop.baidu.com/server_api",
                data=body,
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                result = json.loads(response.read().decode())

            if result.get('err_msg') == 'success.' and 'result' in result:
                return result['result'][0].strip() if result['result'] else ""
            else:
                error_msg = result.get('err_msg', '未知错误')
                raise RuntimeError(f"百度识别失败: {error_msg}")

        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            raise RuntimeError(f"百度 API 调用异常: {e}")

    def listen_for_speech(self, timeout: Optional[int] = None) -> str:
        """
        监听语音输入（带完整异常处理）

        参数:
            timeout: 超时时间（秒），如果为None则使用配置中的值

        返回:
            识别的文本字符串（可能为空）
        """
        actual_timeout = timeout or self.config['timeout']

        try:
            with sr.Microphone() as source:
                print("🎤 请回答（说完后稍作停顿即可，系统会自动识别结束）...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.8)

                audio = self.recognizer.listen(source, timeout=actual_timeout)
                print("⏳ 正在识别语音...")

                text = self.recognize_speech_baidu(audio)
                if text:
                    print(f"📝 你说的是: '{text}'")
                else:
                    print("📝 未识别到有效语音内容")
                return text

        except sr.WaitTimeoutError:
            print(f"⚠️ 超时：{actual_timeout}秒内未检测到语音")
            return ""
        except sr.UnknownValueError:
            print("⚠️ 无法理解语音内容")
            return ""
        except (RuntimeError, ValueError) as e:
            print(f"❌ 语音识别失败: {e}")
            return ""
        except Exception as e:
            print(f"❌ 未知错误: {e}")
            return ""