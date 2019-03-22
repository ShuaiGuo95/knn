iris = csvread('iris.csv');
iris(:, 1:4) = zscore(iris(:, 1:4)); % ���ݱ�׼����

n = size(iris, 1); % ����������
rng(0); % �̶���������ӣ�
randnum = randperm(n); % ������ң�
tr = iris(randnum(1:n*2/3), :); % ȡǰ2/3Ϊѵ������
te = iris(randnum(n*2/3+1:n), :); % ȡ��1/3Ϊ���Ի���

k = 5; % kֵ��
X_tr = tr(:, 1:4); y_tr = tr(:, 5); % �����������ǩ��
X_te = te(:, 1:4); y_te = te(:, 5);

metrics = pdist2(X_te, X_tr); % ���������
[dis, ind] = sort(metrics, 2); % ��������

neighbors = y_tr(ind); % �����ھӵı�ǩ��
knearest_neighbors = neighbors(:, 1:k); % k���ڵı�ǩ��

y_te_pr = mode(knearest_neighbors, 2); % �������ı�ǩ��
accuracy = sum(y_te_pr==y_te)/size(y_te, 1) %׼ȷ�ʼ��㣻