for domain in domain0 domain1 domain2 domain3 domain4 domain5 domain6 domain7
do
    CUDA_VISIBLE_DEVICES=1 python get_score.py 'llama3.2-3b' $domain 1
done