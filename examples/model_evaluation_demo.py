"""Toy data only. This demo does NOT replay historical NCMA measurements."""

from pprint import pprint

from validation_patterns.asr_benchmark import SpeechSample, evaluate_asr
from validation_patterns.llm_benchmark import InferenceSample, evaluate_llm
from validation_patterns.split_audit import (
    DevCheckpoint, Example, Split, audit_splits, select_checkpoint,
)


def main() -> None:
    print('SYNTHETIC speech evaluation')
    samples = [SpeechSample('s1', 'open the window', 'open the window', 3.0, 0.11),
               SpeechSample('s2', 'start the application', 'start application', 4.0, 0.12),
               SpeechSample('s3', 'close the dialog', 'close the dialog', 3.5, 0.09)]
    pprint(evaluate_asr(samples).as_dict())

    print('\nSYNTHETIC LLM inference evaluation')
    tokens = [InferenceSample('text-1', 100, 51, 0.2, 1.2, 'stop'),
              InferenceSample('text-2', 150, 80, 0.3, 2.6, 'length')]
    pprint(evaluate_llm(tokens).as_dict())

    print('\nSYNTHETIC split audit and dev-only checkpoint selection')
    rows = [Example('a', Split.TRAIN, 'group-a', 'one'),
            Example('b', Split.DEV, 'group-b', 'two'),
            Example('c', Split.HOLDOUT, 'group-c', 'three')]
    print('Split status:', audit_splits(rows).status.value)
    choice = select_checkpoint(
        [DevCheckpoint(1, 18, 20, 0.10, 0.02),
         DevCheckpoint(2, 20, 20, 0.0, 0.03),
         DevCheckpoint(3, 20, 20, 0.0, 0.03)],
        max_general_wer=0.05,
    )
    print('Selected epoch (no holdout inspected):', choice.epoch)


if __name__ == '__main__':
    main()
