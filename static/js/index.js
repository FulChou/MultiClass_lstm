var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var _React = React,
    Component = _React.Component;
var _ReactDOM = ReactDOM,
    render = _ReactDOM.render;

var TextLoader = function (_Component) {
  _inherits(TextLoader, _Component);

  function TextLoader() {
    var _ref;

    var _temp, _this, _ret;

    _classCallCheck(this, TextLoader);

    for (var _len = arguments.length, args = Array(_len), _key = 0; _key < _len; _key++) {
      args[_key] = arguments[_key];
    }

    return _ret = (_temp = (_this = _possibleConstructorReturn(this, (_ref = TextLoader.__proto__ || Object.getPrototypeOf(TextLoader)).call.apply(_ref, [this].concat(args))), _this), _this.state = {
      lettersCount: 0,
      inputValue: '',
      countLimit: _this.props.limit,
      loaderCompleted: 0,
      errorLimit: false
    }, _this.handleType = function (e) {
      var val = e.target.value;
      if (val.length <= _this.state.countLimit) {
        _this.setState(function (prev) {
          return {
            lettersCount: val.length,
            inputValue: val,
            errorLimit: false,
            loaderCompleted: Math.floor(val.length / prev.countLimit * 100, 2)
          };
        });
      } else {
        _this.setState(function (prev) {
          return {
            inputValue: prev.inputValue,
            errorLimit: true
          };
        });
      }
    }, _temp), _possibleConstructorReturn(_this, _ret);
  }

  _createClass(TextLoader, [{
    key: 'render',
    value: function render() {
      var _state = this.state,
          inputValue = _state.inputValue,
          lettersCount = _state.lettersCount,
          errorLimit = _state.errorLimit,
          loaderCompleted = _state.loaderCompleted,
          countLimit = _state.countLimit;
      var _props = this.props,
          loaderColor = _props.loaderColor,
          inputType = _props.inputType;

      var loaderStyles = {
        display: lettersCount === 0 ? 'none' : 'block',
        borderColor: loaderColor,
        boxShadow: '0 0 7px ' + loaderColor,
        width: loaderCompleted + '%'
      };
      return React.createElement(
        'div',
        { className: 'loader-wrapper animated ' + (errorLimit ? 'shake' : '') },
        React.createElement(
          'div',
          { className: 'input-wrapper' },
          inputType !== 'textarea' ? React.createElement('input', { id:'input', type: inputType, onChange: this.handleType, value: inputValue, placeholder: 'write some thing here..' }) : React.createElement('textarea', { type: inputType, onChange: this.handleType, value: inputValue, placeholder: 'write some thing here..' }),
          React.createElement('span', { className: 'loader', style: loaderStyles })
        ),
        React.createElement(
          'div',
          { className: '' },
          React.createElement(
            'span',
            { className: 'count' },
            lettersCount,
            '/',
            countLimit
          )
        )
      );
    }
  }]);

  return TextLoader;
}(Component);

TextLoader.propTypes = {
  loaderColor: React.PropTypes.string,
  limit: React.PropTypes.string,
  inputType: React.PropTypes.string.isRequired
};

render(
//change loader color & letters limits by your choice
// inputType coulde be 'number','email','text' & 'textarea'
React.createElement(TextLoader, {
  loaderColor: 'orange',
  limit: '300',
  inputType: 'text' }), document.querySelector('#root')
);