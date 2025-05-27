# 正确设置CUDAVISIBLEDEVICES环境变量
export CUDAVISIBLEDEVICES=1,2

numcount=$(expr length "$CUDAVISIBLEDEVICES" / 2 + 1)

echo "字符串的长度为: $numcount"