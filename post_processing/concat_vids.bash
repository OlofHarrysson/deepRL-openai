dirn=tmp
mkdir $dirn

for i in ./videos/*.mp4; do
    [ -f "$i" ] || break
    fname=$(basename $i) # Filename
    frame_hold=0.3

    # Re-encode renders
    ffmpeg -i $i -codec:v h264 -pix_fmt yuv420p $dirn/$fname

    # get first image of render as img
    ffmpeg -i $dirn/$fname -vf "select=eq(n\,0)" -q:v 3 $dirn/frame_$fname.png

    # take render and make it into a mp4
    ffmpeg  -loop 1 -i $dirn/frame_$fname.png -codec:v h264 -t $frame_hold -pix_fmt yuv420p $dirn/header_$fname

    # build mylist.txt file
    echo "file header_$fname" >> $dirn/concat_list.txt
    echo "file $fname" >> $dirn/concat_list.txt
done

# concat videos
ffmpeg -f concat -safe 0 -i $dirn/concat_list.txt -c copy $dirn/output.mp4

cp ./$dirn/output.mp4 .
rm -r ./tmp