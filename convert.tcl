# code written by my father (Pascal Martin) in TCL that partially reformats formats-data.ts into a json file (incomplete).
# For more information see "Scrape_and_Wrangling.ipynb" in the "Tiering Data" section

set f [open ~/Downloads/formats-data.json r]

set before ""
while {![eof $f]} {
    set line [gets $f]
    if {[string match {\}[,;]} [string trim $line]]} {
        puts [string trim $before {,}]
    } else {
        puts $before
    }
    set before [string trim $line {;}]
}
puts [string trim $line {;}]

