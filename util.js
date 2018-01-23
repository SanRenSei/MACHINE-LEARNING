

function containsNaN(arr) {
  var i;
  for (i=0;i<arr.length;i++) {
    if (isNaN(arr[i]) || (arr[i]==null)) {
      return true;
    }
  }
  return false;
}

exports.containsNaN = containsNaN;