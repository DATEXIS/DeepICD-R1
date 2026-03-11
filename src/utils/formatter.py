from dataclasses import dataclass
from functools import cached_property


@dataclass
class MatchWithTraceAndQuery:
    trace: list[str]
    query: list[str]
    prediction: str = None
    ground_truth: str = None
    overlap: int = None

    @cached_property
    def title(self):
        return f'{self.query} -> {self.prediction} -> {self.ground_truth}'
    
@dataclass
class MatchWithTraceAndPrediction:
    trace: list[str]
    prediction: str = None
    ground_truth: str = None
    overlap: int = None

    @cached_property
    def title(self):
        return f'{self.prediction} -> {self.ground_truth}'


def format_match(match: MatchWithTraceAndQuery) -> tuple[str | None, str | None]:
    if match.overlap is not None:
        if match.ground_truth is match.prediction:
            return _green(match.prediction, match.title), _green(match.ground_truth, match.title)
        else:
            return (_partial_green(match.prediction, match.overlap, match.title),
                    _partial_green(match.ground_truth, match.overlap, match.title))
    return (_red(match.prediction, match.title) if match.prediction is not None else None,
            _red(match.ground_truth, match.title) if match.ground_truth is not None else None)


def format_diagnosis(query_match: str, matches: list[MatchWithTraceAndQuery]) -> str:
    for match in matches:
        for query in match.query:
            if query_match == query:
                if match.overlap is not None:
                    if match.ground_truth is match.prediction:
                        return _green(query_match, match.title)
                    else:
                        return _orange(query_match, match.title)
                elif match.prediction is None:
                    return _blue(query_match, match.title)
                else:
                    return _red(query_match, match.title)
    return query_match


def format_predictions(icd_code: str, matches: list[MatchWithTraceAndQuery]) -> str:
    return _format_pred_or_gt(icd_code, matches, 'prediction')


def format_ground_truth(ground_truth: str, matches: list[MatchWithTraceAndQuery]) -> str:
    return _format_pred_or_gt(ground_truth, matches, 'ground_truth')


def _format_pred_or_gt(string: str, matches: list[MatchWithTraceAndQuery], pred_or_gt_attribute: str) -> str:
    for match in matches:
        if string == getattr(match, pred_or_gt_attribute):
            if match.ground_truth is match.prediction:
                return _green(string, match.title)
            elif match.overlap is not None:
                return _partial_green(string, match.overlap, match.title)
            else:
                return _red(string, match.title)
    return string


def _partial_green(string: str, overlap: int, title: str) -> str:
    return _green(string[:overlap], title) + _orange(string[overlap:], title)


def _green(string: str, title: str) -> str:
    return _span(string, 'rgb(0, 163, 104)', title)


def _orange(string: str, title: str) -> str:
    return _span(string, 'rgb(230, 145, 56)', title)


def _red(string: str, title: str) -> str:
    return _span(string, 'rgb(255, 56, 89)', title)


def _blue(string: str, title: str) -> str:
    return _span(string, 'rgb(51, 141, 216)', title)


def _span(string: str, color: str, title: str = ''):
    return f'<span title="{title}" style="color: {color};">{string}</span>'
