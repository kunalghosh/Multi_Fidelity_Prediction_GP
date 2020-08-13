#!/bin/bash

curl -o HOMO.txt https://zenodo.org/record/3967308/files/HOMO.txt?download=1 
curl -o mbtr_k2.npz https://zenodo.org/record/3967308/files/mbtr_k2.npz?download=1

echo "\nChecking if the files downloaded correctly..\n"

uname0=`uname`

if [[ "$uname0" == "Darwin" ]]; then
   # echo "Using Mac MD5 function"
   function md5sum {
        md5 $1 | cut -d " " -f 4
       }
elif [[ "$uname0" == "Linux" ]]; then
   # echo "Using Linux MD5 function"
   function md5sum {
        md5sum $1 | cut -d " " -f 0
       }
fi

md5_homo_file=`md5sum HOMO.txt`
md5_mbtrk2_file=`md5sum mbtr_k2.npz`

# echo "HOMO's md5 $md5_homo_file"
# echo "MBTR_K2's md5 $md5_mbtrk2_file"

md5_homo=747150277a47d576b461970e45c3bae6 #homo
md5_mbtrk2=7b600c777d92767849005185d01a12a1 #mbtrk2

if [[ "$md5_homo_file" == "$md5_homo" ]]; then
    echo "HOMO.txt downloaded correctly"
else
    echo "HOMO.txt downloaded incorrectly, MD5 check failed"
fi

if [[ "$md5_mbtrk2_file" == "$md5_mbtrk2" ]]; then
    echo "MBTR_k2.npz downloaded correctly"
else
    echo "MBTR_k2.npz downloaded incorrectly, MD5 check failed"
fi
    
