import numpy as np

# Реализуем класс узла

class Node:

    def __init__(self, index, t, true_branch, false_branch):
        self.index = index  # индекс признака, по которому ведется сравнение с порогом в этом узле
        self.t = t  # значение порога
        self.true_branch = true_branch  # поддерево, удовлетворяющее условию в узле
        self.false_branch = false_branch  # поддерево, не удовлетворяющее условию в узле


# И класс терминального узла (листа)

class Leaf:

    def __init__(self, data, target, tree_type='classification'):
        self.data = data
        self.target = target
        self.tree_type = tree_type
        self.prediction = self.predict()

    def predict(self):
        if self.tree_type == 'regression':
            prediction = np.mean(self.target)
            return prediction
        # подсчет количества объектов разных классов
        classes = {}  # сформируем словарь "класс: количество объектов"
        for label in self.target:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        #  найдем класс, количество объектов которого будет максимальным в этом листе и вернем его
        prediction = max(classes, key=classes.get)
        return prediction


# Расчет критерия Джини

def gini(labels):
    #  подсчет количества объектов разных классов
    classes = {}
    for label in labels:
        if label not in classes:
            classes[label] = 0
        classes[label] += 1

    #  расчет критерия
    impurity = 1
    for label in classes:
        p = classes[label] / len(labels)
        impurity -= p ** 2

    return impurity


# Расчет дисперсии

def variance(target):
    if len(target) <= 30:
        return np.var(target, ddof=1)
    return np.var(target)


# Расчет качества

def quality(left_labels, right_labels, current_gini):

    # доля выбоки, ушедшая в левое поддерево
    p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])

    return current_gini - p * gini(left_labels) - (1 - p) * gini(right_labels)


# Разбиение датасета в узле

def split(data, target, index, t):

    left = np.where(data[:, index] <= t)
    right = np.where(data[:, index] > t)

    true_data = data[left]
    false_data = data[right]
    true_target = target[left]
    false_target = target[right]

    return true_data, false_data, true_target, false_target


# Нахождение наилучшего разбиения

def find_best_split(data, target, tree_type='classification'):

    #  обозначим минимальное количество объектов в узле
    min_leaf = 5

    if tree_type == 'regression':
        current_variance = variance(target)
    else:
        current_gini = gini(target)

    best_quality = 0
    best_t = None
    best_index = None

    n_features = data.shape[1]

    for index in range(n_features):
        t_values = [row[index] for row in data]

        for t in t_values:
            true_data, false_data, true_target, false_target = split(data, target, index, t)
            #  пропускаем разбиения, в которых в узле остается менее 5 объектов
            if len(true_data) < min_leaf or len(false_data) < min_leaf:
                continue

            if tree_type == 'regression':
                current_quality = current_variance - variance(true_target)
            else:
                current_quality = quality(true_target, false_target, current_gini)

            #  выбираем порог, на котором получается максимальный прирост качества
            if current_quality > best_quality:
                best_quality, best_t, best_index = current_quality, t, index

    return best_quality, best_t, best_index


# Построение дерева с помощью рекурсивной функции

def build_tree(data, target, max_depth=None, n_splits=0,
               max_feature_splits=None, feature_splits={},
               tree_type='classification'):

    quality, t, index = find_best_split(data, target, tree_type)

    #  Базовый случай - прекращаем рекурсию, когда нет прироста в качества
    if quality == 0:
        return Leaf(data, target, tree_type)

    #  Ограничение на глубину
    if max_depth:
        if n_splits == max_depth:
            return Leaf(data, target, tree_type)
        n_splits += 1

    # Ограничение на количество разбиений признака
    if max_feature_splits:
        feature_splits[index] = feature_splits.setdefault(index, 0) + 1
        if max_feature_splits in feature_splits.values():
            return Leaf(data, target, tree_type)

    true_data, false_data, true_target, false_target = split(data, target, index, t)

    # Рекурсивно строим два поддерева,
    # передавая обновляемые параметры
    true_branch = build_tree(true_data, true_target, max_depth, n_splits,
                             max_feature_splits, feature_splits, tree_type)
    false_branch = build_tree(false_data, false_target, max_depth, n_splits,
                              max_feature_splits, feature_splits, tree_type)

    # Возвращаем класс узла со всеми поддеревьями, то есть целого дерева
    return Node(index, t, true_branch, false_branch)


def predict_object(obj, node):

    #  Останавливаем рекурсию, если достигли листа
    if isinstance(node, Leaf):
        answer = node.prediction
        return answer

    if obj[node.index] <= node.t:
        return predict_object(obj, node.true_branch)
    else:
        return predict_object(obj, node.false_branch)


def predict(data, tree):

    predictions = []
    for obj in data:
        prediction = predict_object(obj, tree)
        predictions.append(prediction)
    return predictions


# Напечатаем ход нашего дерева
def print_tree(node, spacing=""):

    # Если лист, то выводим его прогноз
    if isinstance(node, Leaf):
        print(spacing + "Прогноз: " + str(node.prediction))
        return

    # Выведем значение индекса и порога на этом узле
    print(spacing + 'Индекс: '+ str(node.index))
    print(spacing + 'Порог: ' + str(node.t))

    # Рекурсионный вызов функции на положительном поддереве
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Рекурсионный вызов функции на положительном поддереве
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


if __name__ == '__main__':
    pass
