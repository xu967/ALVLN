#!/usr/bin/env python3
"""分类指标，口径同 src/eval_metrics.cpp（宏 F1 / 平衡准确率 / 每类召回，行=真值、列=预测）。

没有样本的类：召回记 0；P+R=0 的类 F1 记 0；宏平均对全部类等权（空类也算分母）。
与 scikit-learn 的 f1_score(average='macro', zero_division=0)、balanced_accuracy 的差别是：
sklearn 的 balanced_accuracy 会丢掉空类，这里不丢（与 C++ 一致，val 里没有 SAFE 样本时数字因此偏低）。
"""
import numpy as np


def confusion(truth, pred, classes):
    """truth/pred：整数数组 -> classes×classes 的 int64 混淆矩阵（行=真值，列=预测）。"""
    truth = np.asarray(truth, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    c = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(c, (truth, pred), 1)
    return c


def per_class_recall(conf):
    conf = np.asarray(conf, dtype=np.float64)
    row = conf.sum(axis=1)
    diag = np.diag(conf)
    return np.where(row > 0, diag / np.where(row > 0, row, 1.0), 0.0)


def macro_f1(conf):
    conf = np.asarray(conf, dtype=np.float64)
    row, col, diag = conf.sum(axis=1), conf.sum(axis=0), np.diag(conf)
    recall = np.where(row > 0, diag / np.where(row > 0, row, 1.0), 0.0)
    precision = np.where(col > 0, diag / np.where(col > 0, col, 1.0), 0.0)
    s = precision + recall
    f1 = np.where(s > 0, 2.0 * precision * recall / np.where(s > 0, s, 1.0), 0.0)
    return float(f1.mean())


def balanced_accuracy(conf):
    return float(per_class_recall(conf).mean())


def summarize(conf):
    """给报告用的一组数。"""
    conf = np.asarray(conf, dtype=np.int64)
    return {"macro_f1": macro_f1(conf), "balanced_accuracy": balanced_accuracy(conf),
            "recall": per_class_recall(conf).tolist(), "accuracy": float(np.trace(conf) / max(1, conf.sum())),
            "confusion": conf.reshape(-1).tolist(), "n": int(conf.sum())}
