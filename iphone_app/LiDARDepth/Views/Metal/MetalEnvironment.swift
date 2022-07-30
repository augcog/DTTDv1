/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A singleton object that holds the MTLDevice and MTLCommandQueue for the app.
*/

import Foundation
import Metal

class MetalEnvironment {
    
    static let shared: MetalEnvironment = { MetalEnvironment() }()
    
    let metalDevice: MTLDevice
    let metalCommandQueue: MTLCommandQueue
    let metalLibrary: MTLLibrary
    
    private init() {
        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            fatalError("Unable to create the metal device.")
        }
        guard let metalCommandQueue = metalDevice.makeCommandQueue() else {
            fatalError("Unable to create the command queue.")
        }
        guard let metalLibrary = metalDevice.makeDefaultLibrary() else {
            fatalError("Unable to create the default library.")
        }
        self.metalDevice = metalDevice
        self.metalCommandQueue = metalCommandQueue
        self.metalLibrary = metalLibrary
    }
}

