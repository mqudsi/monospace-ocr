#!/usr/bin/env fish

set generation 000
if set -q argv[1]
	set generation $argv[1]
end

if true
# Train with this permutation of 1/l training data
uv.exe run cluster.py ../EFTA00382108-002.png ./train_top.txt ./train_bot.txt -q -d

# Generate outputs according to what we just trained
parallel -j 16 uv.exe run cluster.py ../EFTA00382108-{}.png -o EFTA00382108-{}.txt ::: (printf "%03d\n" (seq 2 23))

# Remove non-base64 characters from the top of page 2
tail -n +7 < EFTA00382108-002.txt | rewrite EFTA00382108-002.txt

# Remove everything after the single = padding on page 23
string replace -r "(?s)=.*" "=" "$(cat EFTA00382108-023.txt)" | rewrite EFTA00382108-023.txt
end

# Concatenate all and try to generate output
cat EFTA00382108-*.txt | string replace "> " "" \
	| b64-debug/target/release/b64-debug --strict > airmail.pdf

# Save results and training data to a subfolder
set DEST EFTA00382108/$generation
mkdir -p $DEST
base64 < airmail.pdf > $DEST/airmail.base64.txt
mv airmail.pdf $DEST/airmail.pdf
cp train_top.txt train_bot.txt $DEST/

# Check for corruption
set stderr (qpdf $DEST/airmail.pdf $DEST/airmail.rebuilt.pdf 2>&1 1>/dev/null)
if string match -qr '\w' $stderr
	printf "%s\n" -- $stderr > $DEST/qpdf-warnings.txt
	exit 1
end

rm -f $DEST/qpdf-warnings.txt
exit 0
