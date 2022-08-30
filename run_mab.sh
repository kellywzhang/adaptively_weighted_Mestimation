
T=1000
strategy='TS'
load_results=0
reward='normal'
var=1
N=50000

clip=0.01
means=0,1

clip=0.0
means=0,0

adjust=0
estvar=0
awaipw=0
ols=1
awls=1

python mab_simulations.py --strategy $strategy --T $T --means $means --clipping $clip --reward $reward --var $var --N $N

allT="100"
for t in $(seq 200 100 $T);
do
    allT="$allT,$t"
done

python process_mab.py --strategy $strategy --T $T --means $means --clipping $clip --verbose 1 --sparseT $allT --adjust $adjust --reward $reward --estvar $estvar --awaipw $awaipw --ols $ols --load_results $load_results --var $var --N $N --awls $awls


exit 0


