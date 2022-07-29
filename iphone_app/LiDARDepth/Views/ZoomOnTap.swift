/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A view that clones a contained view to full screen when a user taps it.
*/

import Foundation
import SwiftUI

struct ZoomOnTap<T: View>: View {
        
    let content: T
    @State private var isPresented = false

    init(@ViewBuilder content: () -> T) {
        self.content = content()
    }
    
    var body: some View {
        content.onTapGesture { isPresented.toggle() }
            .fullScreenCover(isPresented: $isPresented) {
                content.onTapGesture { isPresented.toggle() }
            }
    }
}

