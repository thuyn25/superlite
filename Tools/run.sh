 3 for ((day=-230; day<=-100; day+=5)); do
  4     #formatted_day=$(printf "%04d" "$day")
  5     #dir="day_${formatted_day}"
  6 #for dir in day_*/; do
  7     dir="${dir%/}"
  8     if [ ! -d "$dir" ]; then
  9         echo "Directory $dir does not exist, skipping..."
 10         continue # Skips the rest of this loop iteration and moves to the next day
 11     fi
 12      echo "$dir"
 13      echo "Running command in $dir"
 14      #cd "$dir" || exit 1
 15      #srun -N1 -n64 ./superlite | tee output.log
 16      #cd .. || exit 1