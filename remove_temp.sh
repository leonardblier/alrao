#!/bin/bash

if [ $# -eq 0 ]
then
    echo "Removes temporary files matching at least one of the conditions given in argument:"
    echo "--all        - remove ALL temporary files"
    echo "--not_kept   - remove temp files which should have been removed"
    echo "--used       - remove used temp files which should have been removed"
    echo "--unused     - remove unused temp files which should have been removed"
    echo "--all_used   - remove ALL used temp files"
    echo "--all_unused - remove ALL unused temp files"
    echo "example: 'bash remove_temp.sh --all_used --not_kept' removes both used temp files and files which should have been removed"
fi

for arg in "$@"
do
    for f in *.sh
    do
        if grep -Fxq "#TEMPORARY_FILE" "$f"
        then
            if [ "$arg" == "--all" ]
            then
                rm $f
            elif [ "$arg" == "--all_unused" ] && ! grep -Fxq "#USED_FILE" "$f"
            then
                rm $f
            elif [ "$arg" == "--all_used" ] && grep -Fxq "#USED_FILE" "$f"
            then
                rm $f
            elif ! grep -Fxq "#KEEP_FILE" "$f"
            then
                if [ "$arg" == "--not_kept" ]
                then
                    rm $f
                elif [ "$arg" == "--unused" ] && ! grep -Fxq "#USED_FILE" "$f"
                then
                    rm $f
                elif [ "$arg" == "--used" ] && grep -Fxq "#USED_FILE" "$f"
                then
                    rm $f
                fi
            fi
        fi
    done
done
