
"use strict";

let SetTorqueLimit = require('./SetTorqueLimit.js')
let SetSpeed = require('./SetSpeed.js')
let SetComplianceMargin = require('./SetComplianceMargin.js')
let StartController = require('./StartController.js')
let RestartController = require('./RestartController.js')
let SetCompliancePunch = require('./SetCompliancePunch.js')
let TorqueEnable = require('./TorqueEnable.js')
let StopController = require('./StopController.js')
let SetComplianceSlope = require('./SetComplianceSlope.js')

module.exports = {
  SetTorqueLimit: SetTorqueLimit,
  SetSpeed: SetSpeed,
  SetComplianceMargin: SetComplianceMargin,
  StartController: StartController,
  RestartController: RestartController,
  SetCompliancePunch: SetCompliancePunch,
  TorqueEnable: TorqueEnable,
  StopController: StopController,
  SetComplianceSlope: SetComplianceSlope,
};
