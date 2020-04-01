// pages/chat/chat.js
const app = getApp()
var utils = require('../../utils/util.js');
Page({

  /**
   * 页面的初始数据
   */
  data: {
     newslist:[],
     userInfo: {},
     scrollTop: 0,
     message:""
  },
  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function () {
    var that = this
    if (app.globalData.userInfo) {
      this.setData({
        userInfo: app.globalData.userInfo
      })
    }
  },
  //事件处理函数
  send: function () {
    var that = this
    if (this.data.message.trim() == ""){
      wx.showToast({
        title: '消息不能为空哦~',
        icon: "none",
        duration: 2000
      })
    } else {
      var list = that.data.newslist
      list.push({'nickName': that.data.userInfo.nickName, 'content': this.data.message, 'type': 'text' })
      that.setData({
        newslist: list
      })
      // send http request
      wx.request({
        url: 'http://127.0.0.1:8888/talk?content=' + this.data.message,
        method: "GET",
        header: { "Content-type": "application/json" },
        success: function(res) {
          var data = res.data
          list.push(data)
          that.setData({
            newslist: list
          })
          that.bottom()
        },
        fail: function() {
          var data = { 'nickName': '小AI', 'content': '小AI断网了,不能愉快聊天了~', 'type': 'text' }
          list.push(data)
          that.setData({
            newslist: list
          })
          that.bottom()
        }
      })
    }
  },
  //监听input值的改变
  bindChange(res) {
    this.setData({
      message : res.detail.value
    })
  },
  cleanInput(){
    //button会自动清空，所以不能再次清空而是应该给他设置目前的input值
    this.setData({
      message: this.data.message
    })
  },
  //聊天消息始终显示最底端
  bottom: function () {
    var query = wx.createSelectorQuery()
    query.select('#flag').boundingClientRect()
    query.selectViewport().scrollOffset()
    query.exec(function (res) {
      wx.pageScrollTo({
        scrollTop: res[0].bottom  // #the-id节点的下边界坐标  
      })
      res[1].scrollTop // 显示区域的竖直滚动位置  
    })
  },  
})