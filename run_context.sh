
T=1000
sparseT='200,400,600,800,1000'
N=5000

clip=0.05
strategy='TS'
d=3
params="0.1,0.1,0.1,0,0,0"
#params="0.1,0.1,0.1,0.2,0.1,0"
estvar=1
path='./context_simulations'
Z_dist='uniform'

load_results=0
Wdecor=1
snmb=1

for reward in 't-dist' 'bernoulli' 'poisson'
do
    echo $reward
    echo python context_simulations.py --strategy $strategy --T $T --clipping $clip --reward $reward --N $N --d $d --Z_dist $Z_dist --params $params --path $path
    python context_simulations.py --strategy $strategy --T $T --clipping $clip --reward $reward --N $N --d $d --Z_dist $Z_dist --params $params --path $path

    echo python process_context.py --strategy $strategy --T $T --clipping $clip --reward $reward --N $N --sparseT $sparseT --d $d --estvar $estvar --Z_dist $Z_dist --params $params --path $path --load_results $load_results
    python process_context.py --strategy $strategy --T $T --clipping $clip --reward $reward --N $N --sparseT $sparseT --d $d --estvar $estvar --Z_dist $Z_dist --params $params --path $path --load_results $load_results --Wdecor $Wdecor --snmb $snmb
done

exit 0


