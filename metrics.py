import jiwer
from arabic import main

ground_truth_dir = 'data/Groundtruth-Unicode.xlsx'

ground_truth = list(ground_truth_dir)
hypothesis = [main.predicted]

wer = jiwer.wer(ground_truth, hypothesis)
print("word error rate (WER): %f", wer)
#mer = jiwer.mer(ground_truth, hypothesis)
#wil = jiwer.wil(ground_truth, hypothesis)
