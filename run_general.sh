for att_key in {gaussian,lf}
do

for epsilon in {2,1,0.5,0.25,0.125}
do

for start_att in 0.0
do

for mal_worker_portion in {60,40,20}
do

for seed in {1,2,3}
do

for anti_byz in 1
do

for non_iid in 0
do

for base_lr in 0.2
do

python main.py --dataset mnist      --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset colorectal --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset fashion    --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset usps       --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr

done
done
done
done
done
done
done
done

for att_key in local
do

for epsilon in {2,1,0.5,0.25,0.125}
do

for start_att in 0.0
do

for mal_worker_portion in {60,40,20}
do

for seed in {1,2,3}
do

for anti_byz in 1
do

for non_iid in 0
do

for base_lr in 0.2
do

python main.py --dataset mnist      --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset colorectal --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset fashion    --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset usps       --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr

done
done
done
done
done
done
done
done

for att_key in nobyz
do

for epsilon in {2,1,0.5,0.25,0.125}
do

for start_att in 0
do

for mal_worker_portion in 20
do

for seed in {1,2,3}
do

for anti_byz in 0
do

for non_iid in 0
do

for base_lr in 0.2
do

python main.py --dataset mnist      --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset colorectal --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset fashion    --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset usps       --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr

done
done
done
done
done
done
done
done





for att_key in {gaussian,lf}
do

for epsilon in {2,1,0.5,0.25,0.125}
do

for start_att in 0.0
do

for mal_worker_portion in {60,40,20}
do

for seed in {1,2,3}
do

for anti_byz in 1
do

for non_iid in 1
do

for base_lr in 0.2
do

python main.py --dataset mnist      --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset colorectal --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset fashion    --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset usps       --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr

done
done
done
done
done
done
done
done

for att_key in local
do

for epsilon in {2,1,0.5,0.25,0.125}
do

for start_att in 0.0
do

for mal_worker_portion in {60,40,20}
do

for seed in {1,2,3}
do

for anti_byz in 1
do

for non_iid in 1
do

for base_lr in 0.2
do

python main.py --dataset mnist      --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset colorectal --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset fashion    --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset usps       --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr

done
done
done
done
done
done
done
done

for att_key in nobyz
do

for epsilon in {2,1,0.5,0.25,0.125}
do

for start_att in 0
do

for mal_worker_portion in 20
do

for seed in {1,2,3}
do

for anti_byz in 0
do

for non_iid in 1
do

for base_lr in 0.2
do

python main.py --dataset mnist      --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset colorectal --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset fashion    --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr
python main.py --dataset usps       --att_key $att_key --epsilon $epsilon --DP_mode localDP --seed $seed --mal_worker_portion $mal_worker_portion --anti_byz $anti_byz --non_iid $non_iid --start_att $start_att --base_lr $base_lr

done
done
done
done
done
done
done
done


