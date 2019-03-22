root = dir('./face19/'); % ��Ŀ¼��
data = []; 
for i = 3:length(root)
    foldi_index = ['./face19/', root(i).name, '/']; % ��i������ͼƬ��Ŀ¼��
    imagesi = dir([foldi_index, '*.jpg']); % ����fold_i������jpgͼƬ��
    for j = 1:length(imagesi)
        imageij_index = [foldi_index, imagesi(j).name];
        imageij = rgb2gray(imread(imageij_index)); % ����ͼƬ��ת��Ϊ�Ҷ�ͼƬ��
        imageij = reshape(imageij, 1, size(imageij, 1)*size(imageij, 2)); % ʸ����Ϊһά���� 1*��������
        imageij = [imageij, i-2];
        data = [data; imageij];
    end
end

n = size(data, 1);
rng(0);
randnum = randperm(n); 
tr = data(randnum(1:n*2/3), :);
te = data(randnum(n*2/3+1:n), :);
k = 5;
X_tr = tr(:, 1:36000); y_tr = tr(:, 36001);
X_te = te(:, 1:36000); y_te = te(:, 36001);

metrics = pdist2(X_te, X_tr); % ���������
[dis, ind] = sort(metrics, 2); % ��������

neighbors = y_tr(ind); % �����ھӵı�ǩ��
knearest_neighbors = neighbors(:, 1:k); % k���ڵı�ǩ��

y_te_pr = mode(knearest_neighbors, 2); % �������ı�ǩ��
accuracy = sum(y_te_pr==y_te)/size(y_te, 1) %׼ȷ�ʼ��㣻
