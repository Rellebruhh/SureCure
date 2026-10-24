package com.mlbb.surecure.rules

/**
 * Step 1 of meta.json's decision_order: "barcode found in product table -> use it, skip photo".
 *
 * Returns the product for a scanned barcode, or null when the code fails the EAN-13 check digit
 * or isn't in barcodes.csv; then the app runs the photo model as usual. The barcode still goes
 * into ScanInput, so verify() applies the same barcode rules either way.
 */
fun FdaTables.productForBarcode(code: String?): String? =
    if (ean13Valid(code)) barcodes[code] else null
