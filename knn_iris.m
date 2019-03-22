iris = csvread('iris.csv');
iris(:, 1:4) = zscore(iris(:, 1:4)); % 数据标准化；

n = size(iris, 1); % 样本总量；
rng(0); % 固定随机数种子；
randnum = randperm(n); % 随机打乱；
tr = iris(randnum(1:n*2/3), :); % 取前2/3为训练集；
te = iris(randnum(n*2/3+1:n), :); % 取后1/3为测试机；

k = 5; % k值；
X_tr = tr(:, 1:4); y_tr = tr(:, 5); % 区分特征与标签；
X_te = te(:, 1:4); y_te = te(:, 5);

metrics = pdist2(X_te, X_tr); % 距离度量；
[dis, ind] = sort(metrics, 2); % 按行排序；

neighbors = y_tr(ind); % 所有邻居的标签；
knearest_neighbors = neighbors(:, 1:k); % k近邻的标签；

y_te_pr = mode(knearest_neighbors, 2); % 出现最多的标签；
accuracy = sum(y_te_pr==y_te)/size(y_te, 1) %准确率计算；