<?php

// CONSTANTS //

// 0=chest, 1=waist, 2=seat/hips, 3=neckband, 4=shirt sleeve, 5=inseam
$sizing_table_in = json_decode("[
    [34.0, 36.0, 38.0, 41.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60], 
    [28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0], 
    [33.0, 35.0, 37.0, 39.0, 41.0, 43.0, 45.0, 47.0, 49.0, 51.0, 53.0, 55.0, 57.0, 59.0], 
    [14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5], 
    [32.0, 33.0, 33.0, 34.0, 34.0, 35.0, 35.0, 36.0, 36.0, 37.0, 37.0, 37.5, 38.0, 38.5], 
    [30.0, 30.0, 31.0, 31.0, 32.0, 32.0, 32.0, 32.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0]
]
");
$sizing_table = json_decode("[
    [86.49940, 91.58760, 96.67580, 104.30810, 106.85220, 111.94040, 117.02860, 122.11680, 127.20500, 132.29320, 137.38140, 142.46960, 147.55780, 152.6460],
    [71.23480, 76.32300, 81.41120, 86.49940, 91.58760, 96.67580, 101.76400, 106.85220, 111.94040, 117.02860, 122.11680, 127.20500, 132.29320, 137.38140], 
    [83.95530, 89.04350, 94.13170, 99.21990, 104.30810, 109.39630, 114.48450, 119.57270, 124.66090, 129.74910, 134.83730, 139.92550, 145.01370, 150.10190], 
    [35.61740, 36.88945, 38.16150, 39.43355, 40.70560, 41.97765, 43.24970, 44.52175, 45.79380, 47.06585, 48.33790, 49.60995, 50.88200, 52.15405], 
    [81.41120, 83.95530, 83.95530, 86.49940, 86.49940, 89.04350, 89.04350, 91.58760, 91.58760, 94.13170, 94.13170, 95.40375, 96.67580, 97.94785], 
    [76.32300, 76.32300, 78.86710, 78.86710, 81.41120, 81.41120, 81.41120, 81.41120, 86.49940, 86.49940, 86.49940, 86.49940, 86.49940, 86.49940]
]");

$cm_per_inch = 2.5441;
$kg_per_lb = 0.453592;

// FUNCTIONS //

// $xp must be monotonically increasing
function linear_interpolate($x, $xp, $yp) {
    $n = count($xp);
    assert(count($xp) == count($yp));
    if ( $x < $xp[0] ) {
        return $yp[0];
    }
    if ( $x > $xp[$n-1] ) {
        return $yp[$n-1];
    }
    for ( $i = 0; $i < $n-1; $i++ ) {
        if ( $x >= $xp[$i] && $x <= $xp[$i+1] ) {
            return ($x - $xp[$i])/($xp[$i+1] - $xp[$i]) * ($yp[$i+1] - $yp[$i]) + $yp[$i];
        }
    }
    return 0;
}

function avg($a) {
    return array_sum($a) / count($a);
}

// INPUTS //

$height_cm = 178; // cm
$weight_kg = 82; // kg
$waist_cm = 86; // cm

$height_cm = $_REQUEST['height'];
$weight_kg = $_REQUEST['weight'];
$waist_cm = $_REQUEST['waist'];

$head_height = $height_cm * 1/8;
$body_length = $height_cm * 7/8;
$shirt_length = $body_length * 1/2;
$shoulder_length = $shirt_length * 5/8;
$ideal_weight_kg = $height_cm - 100;

if ( $weight_kg > $ideal_weight_kg ) {
    $shoulder = $shoulder_length + 0.5 * $cm_per_inch;
} elseif ( $weight_kg < $ideal_weight_kg ) {
    $shoulder = $shoulder_length - 0.25 * $cm_per_inch;
} else {
    $shoulder = $shoulder_length;
}

$long_sleeve = ($height_cm-$shoulder)/2.0;
$short_sleeve = $long_sleeve/2.5;
$shirt_sleeve = $shoulder_length/2.0 + $long_sleeve;

$standard_weight = 0;
$standard_weight_deviation = 0.17;

if ( !$waist_cm ) {
    $waist_cm = linear_interpolate($shirt_sleeve, $sizing_table[4], $sizing_table[2]);
}

$measurements = array(
    'long_sleeve' => $long_sleeve,
    'short_sleeve' => $short_sleeve,
    'neck' => linear_interpolate($shirt_sleeve, $sizing_table[4], $sizing_table[3]),
    'waist' => $waist_cm,
    'chest' => linear_interpolate($shirt_sleeve, $sizing_table[4], $sizing_table[0]),
    'length' => $shirt_length,
    'shoulder' => $shoulder,
    'hip' => linear_interpolate($shirt_sleeve, $sizing_table[4], $sizing_table[2]),
    'shirt_sleeve' => $shirt_sleeve
);

// override neck if they provide it
if ( $_REQUEST['neck'] ) {
    $measurements['neck'] = $_REQUEST['neck'];
}

$real_waist = $measurements['waist']; // do not do correction - 5.08;
if ( $_REQUEST['version'] == 2 ) {

    $measurements['chest'] = avg(array(linear_interpolate($shirt_sleeve, $sizing_table[4], $sizing_table[0]),
                                       linear_interpolate($measurements['neck'], $sizing_table[3], $sizing_table[0]),
                                       linear_interpolate($real_waist, $sizing_table[1], $sizing_table[0])));
    $measurements['hip'] = avg(array(linear_interpolate($shirt_sleeve, $sizing_table[4], $sizing_table[2]),
                                       linear_interpolate($measurements['neck'], $sizing_table[3], $sizing_table[2]),
                                       linear_interpolate($real_waist, $sizing_table[1], $sizing_table[2])));
} 
if ( $_REQUEST['version'] == 3 ) {
    $measurements['chest'] = $height_cm/2.0 + ($height_cm/2.0)/30.0 + 3.0;
    $measurements['hip'] = avg(array(linear_interpolate($shirt_sleeve, $sizing_table[4], $sizing_table[2]),
                                       linear_interpolate($measurements['neck'], $sizing_table[3], $sizing_table[2]),
                                       linear_interpolate($real_waist, $sizing_table[1], $sizing_table[2])));
}

if ( $_REQUEST['callback'] ) {
    header('Content-Type: text/javascript');
    print $_REQUEST['callback'] + '(' + json_encode('') + ');';
} else {
    header('Content-Type: application/json');
    print json_encode($measurements);
}
exit;
?>