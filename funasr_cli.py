import os
import subprocess
import threading
import concurrent.futures
import logging
import argparse
from pathlib import Path
import time
import funasr
from funasr import AutoModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, input_path, output_dir, model_name="paraformer-zh", sample_rate=16000, 
                 mono=True, max_workers=4, keep_wav=False):
        """
        初始化音频处理器
        
        参数:
            input_path (str): 输入文件或目录路径
            output_dir (str): 输出目录路径
            model_name (str): FunASR模型名称
            sample_rate (int): 输出音频采样率
            mono (bool): 是否转换为单声道
            max_workers (int): 最大线程数
            keep_wav (bool): 是否保留中间WAV文件
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.mono = mono
        self.max_workers = max_workers
        self.keep_wav = keep_wav
        
        # 创建输出目录结构
        self.wav_output_dir = self.output_dir / "wav_files"
        self.txt_output_dir = self.output_dir / "txt_files"
        self._create_dirs()

    def _create_dirs(self):
        """创建必要的输出目录"""
        try:
            self.wav_output_dir.mkdir(parents=True, exist_ok=True)
            self.txt_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"输出目录已创建: {self.output_dir}")
        except OSError as e:
            logger.error(f"无法创建目录: {e}")
            raise

    def _load_model(self):
        """加载FunASR语音识别模型"""
        try:
            logger.info(f"正在加载FunASR模型: {self.model_name}")
            self.model = AutoModel(model=self.model_name)
            logger.info("FunASR模型加载完成")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise

    def _get_output_path(self, input_file, output_dir, suffix):
        """
        获取输出路径，处理文件名冲突
        
        参数:
            input_file (Path): 输入文件路径
            output_dir (Path): 输出目录
            suffix (str): 文件后缀
            
        返回:
            Path: 处理后的输出路径
        """
        output_file = output_dir / f"{input_file.stem}{suffix}"
        counter = 1
        while output_file.exists():
            output_file = output_dir / f"{input_file.stem}_{counter}{suffix}"
            counter += 1
            logger.warning(f"文件名称冲突，已重命名为: {output_file.name}")
        return output_file

    def _convert_to_wav(self, input_file):
        """使用FFmpeg将文件转换为WAV格式"""
        try:
            # 检查输入文件是否存在
            if not input_file.exists():
                logger.error(f"输入文件不存在: {input_file}")
                return None
            
            # 处理输出文件名冲突
            wav_file = self._get_output_path(input_file, self.wav_output_dir, ".wav")
            
            # 构建FFmpeg命令
            cmd = ['ffmpeg', '-i', str(input_file)]
            if self.mono:
                cmd.extend(['-ac', '1'])
            cmd.extend([
                '-ar', str(self.sample_rate),
                '-y',  # 覆盖输出文件
                str(wav_file)
            ])
            
            logger.info(f"开始转换: {input_file} -> {wav_file}")
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode == 0:
                logger.info(f"转换成功: {input_file} -> {wav_file}")
                return wav_file
            else:
                logger.error(f"FFmpeg转换失败: {input_file} - {result.stderr.decode()}")
                return None
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg执行错误: {input_file} - {e.stderr.decode()}")
        except Exception as e:
            logger.error(f"转换时发生未知错误: {input_file} - {str(e)}")
        return None

    def _transcribe_audio(self, wav_file):
        if not hasattr(self,"model"):
            # 加载FunASR模型
            self._load_model()

        """使用FunASR转换音频为文本"""
        try:
            # 处理输出文件名冲突
            txt_file = self._get_output_path(wav_file, self.txt_output_dir, ".txt")
            
            logger.info(f"开始语音识别: {wav_file}")
            start_time = time.time()
            
            # 使用FunASR进行语音识别
            res = self.model.generate(input=str(wav_file))
            text = res[0]["text"]
            
            # 写入文本文件
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            processing_time = time.time() - start_time
            logger.info(f"语音识别完成: {wav_file} -> {txt_file} (耗时: {processing_time:.2f}s)")
            
            # 如果不保留WAV文件，则删除
            if not self.keep_wav:
                try:
                    wav_file.unlink()
                    logger.info(f"已删除临时WAV文件: {wav_file}")
                except Exception as e:
                    logger.warning(f"删除WAV文件失败: {wav_file} - {str(e)}")
            
            return txt_file
        except Exception as e:
            logger.error(f"语音识别失败: {wav_file} - {str(e)}")
            return None

    def process_file(self, input_file):
        """处理单个文件：转换格式并转为文本"""
        try:
            wav_file = self._convert_to_wav(input_file)
            if not wav_file:
                return False
            
            txt_file = self._transcribe_audio(wav_file)
            return bool(txt_file)
        except Exception as e:
            logger.error(f"处理文件时出错: {input_file} - {str(e)}")
            return False

    def _collect_audio_files(self):
        """收集所有支持的音频文件"""
        supported_extensions = ['.mp3', '.wav', '.aac', '.flac', '.m4a', '.ogg']
        
        if self.input_path.is_file():
            if self.input_path.suffix.lower() in supported_extensions:
                return [self.input_path]
            else:
                logger.warning(f"不支持的音频格式: {self.input_path}")
                return []
        elif self.input_path.is_dir():
            audio_files = []
            for ext in supported_extensions:
                audio_files.extend(self.input_path.rglob(f"*{ext}"))
            return audio_files
        else:
            logger.error(f"无效的输入路径: {self.input_path}")
            return []

    def run(self):
        """运行音频处理流程"""
        # 收集音频文件
        audio_files = self._collect_audio_files()
        if not audio_files:
            logger.warning("没有找到支持的音频文件")
            return False
        
        logger.info(f"发现 {len(audio_files)} 个待处理文件")
        
        # 使用线程池处理
        success_count = 0
        failed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, file): file 
                for file in audio_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"处理过程中出现异常: {file} - {str(e)}")
                    failed_count += 1
        
        logger.info(f"处理完成: 成功 {success_count} 个, 失败 {failed_count} 个")
        return success_count > 0


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="音频文件处理工具: 转换格式并转录为文本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input", 
        required=True,
        help="输入文件或目录路径"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True,
        help="输出目录路径"
    )
    parser.add_argument(
        "-m", "--model", 
        default="paraformer-zh",
        help="FunASR模型名称"
    )
    parser.add_argument(
        "-r", "--sample-rate", 
        type=int, 
        default=16000,
        help="输出音频采样率"
    )
    parser.add_argument(
        "--no-mono", 
        action="store_false",
        dest="mono",
        help="不转换为单声道"
    )
    parser.add_argument(
        "-w", "--max-workers", 
        type=int, 
        default=4,
        help="最大线程数"
    )
    parser.add_argument(
        "-k", "--keep-wav", 
        action="store_true",
        help="保留中间WAV文件"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志级别
    logging.basicConfig(level=args.log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        processor = AudioProcessor(
            input_path=args.input,
            output_dir=args.output,
            model_name=args.model,
            sample_rate=args.sample_rate,
            mono=args.mono,
            max_workers=args.max_workers,
            keep_wav=args.keep_wav
        )
        
        if not processor.run():
            logger.error("处理过程中出现错误")
            exit(1)
            
    except Exception as e:
        logger.error(f"程序运行失败: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
