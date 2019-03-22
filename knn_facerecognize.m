root = dir('./face19/'); % 根目录；
data = []; 
for i = 3:length(root)
    foldi_index = ['./face19/', root(i).name, '/']; % 第i类人脸图片的目录；
    imagesi = dir([foldi_index, '*.jpg']); % 检索fold_i中所有jpg图片；
    for j = 1:length(imagesi)
        imageij_index = [foldi_index, imagesi(j).name];
        imageij = rgb2gray(imread(imageij_index)); % 读入图片；转化为灰度图片；
        imageij = reshape(imageij, 1, size(imageij, 1)*size(imageij, 2)); % 矢量化为一维向量 1*像素数；
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

metrics = pdist2(X_te, X_tr); % 距离度量；
[dis, ind] = sort(metrics, 2); % 按行排序；

neighbors = y_tr(ind); % 所有邻居的标签；
knearest_neighbors = neighbors(:, 1:k); % k近邻的标签；

y_te_pr = mode(knearest_neighbors, 2); % 出现最多的标签；
accuracy = sum(y_te_pr==y_te)/size(y_te, 1) %准确率计算；
